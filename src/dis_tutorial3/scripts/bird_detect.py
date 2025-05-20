#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
from torchvision import transforms, models
import cv2
import numpy as np
import os
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data

from dis_tutorial3.msg import DetectedBird

class BirdDetector(Node):
    def __init__(self):
        super().__init__('detect_birds')

        self.yolo_model = YOLO("model/bird_yolov8n.pt")
        self.resnet_model_path = "model/bird_species_resnet18.pt"
        self.data_dir = "train_bird_classifier/filtered_data"

        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, 10)
        self.bird_pub = self.create_publisher(DetectedBird, "/detected_birds", 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 24)
        self.resnet.load_state_dict(torch.load(self.resnet_model_path, map_location=self.device))
        self.resnet.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.class_names = sorted(os.listdir(self.data_dir))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_pointcloud = None
        self.pc_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.pc_callback, qos_profile_sensor_data)

        self.groups = []
        self.group_threshold = 0.5
        self.min_detections = 3

        self.get_logger().info("🦜 Bird detection node ready")

    def pc_callback(self, msg):
        self.latest_pointcloud = msg

    def image_callback(self, msg):
        if self.latest_pointcloud is None:
            self.get_logger().warn("No point cloud received yet.")
            return

        img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.yolo_model.predict(source=[img_bgr], conf=0.7, save=False)

        boxes = results[0].boxes

        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = self.transform(transforms.ToPILImage()(crop_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.resnet(crop_tensor)
                probabilities = torch.softmax(pred, dim=1)
                class_confidence, class_idx = torch.max(probabilities, dim=1)
                class_confidence = class_confidence.item()

                #if class_confidence < 0.4:  # Adjust threshold as desired
                #    self.get_logger().info(f"Skipped classification (low confidence: {class_confidence:.2f})")
                #    continue

                class_name = self.class_names[class_idx.item()]
            
            u_center, v_center = (x1 + x2) // 2, (y1 + y2) // 2
            u_range = slice(max(u_center - 2, 0), u_center + 3)
            v_range = slice(max(v_center - 2, 0), v_center + 3)

            pc_array = pc2.read_points_numpy(self.latest_pointcloud, field_names=("x", "y", "z"), skip_nans=True)
            points_reshaped = pc_array.reshape((self.latest_pointcloud.height, self.latest_pointcloud.width, 3))
            points_sample = points_reshaped[v_range, u_range].reshape(-1, 3)

            valid_points = points_sample[np.isfinite(points_sample).all(axis=1) & (np.linalg.norm(points_sample, axis=1) > 0.05)]

            if valid_points.shape[0] < 3:
                self.get_logger().warn("Not enough valid points.")
                continue

            avg_3d = np.mean(valid_points, axis=0)

            stamped_point = PointStamped()
            stamped_point.header.stamp = self.get_clock().now().to_msg()
            stamped_point.header.frame_id = msg.header.frame_id


            stamped_point.point.x = float(avg_3d[0])
            stamped_point.point.y = float(avg_3d[1])
            stamped_point.point.z = float(avg_3d[2])

            
            try:
                transform = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rclpy.time.Time())
                map_point = tf2_geometry_msgs.do_transform_point(stamped_point, transform).point
                self.add_to_group(map_point, class_name)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")

        self.publish_groups()

    def add_to_group(self, point, class_name):
        position = np.array([point.x, point.y, point.z])
        for group in self.groups:
            if np.linalg.norm(group['position'] - position) < self.group_threshold:
                group['positions'].append(position)
                group['classifications'].append(class_name)
                return

        self.groups.append({
            'positions': [position],
            'classifications': [class_name],
            'position': position
        })

    def publish_groups(self):
        for group in self.groups:
            if len(group['positions']) >= self.min_detections:
                avg_pos = np.mean(group['positions'], axis=0)
                most_common_class = max(set(group['classifications']), key=group['classifications'].count)
                confidence = group['classifications'].count(most_common_class) / len(group['classifications'])

                msg = DetectedBird()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.position = Point(x=float(avg_pos[0]), y=float(avg_pos[1]), z=float(avg_pos[2]))
                msg.class_name = most_common_class
                msg.confidence = float(confidence)

                self.bird_pub.publish(msg)
                self.get_logger().info(f"🟢 Published grouped bird: {most_common_class}, Confidence: {confidence:.2f}, Position: {avg_pos.round(2)}")

                self.groups.remove(group)

def main(args=None):
    rclpy.init(args=args)
    node = BirdDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

