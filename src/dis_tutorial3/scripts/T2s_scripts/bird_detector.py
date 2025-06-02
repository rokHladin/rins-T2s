#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header, String
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
import message_filters

from dis_tutorial3.msg import DetectedBird

class BirdDetector(Node):
    def __init__(self):
        super().__init__('detect_birds')

        # Models
        self.yolo_model = YOLO("model/bird_yolov8n.pt")
        self.resnet_model_path = "model/bird_species_resnet18.pt"
        self.data_dir = "train_bird_classifier/filtered_data"

        self.bridge = CvBridge()
        self.bird_pub = self.create_publisher(DetectedBird, "/detected_birds", 10)
        self.init_bird_pub = self.create_publisher(PointStamped, "/bird_initial_position_detection", 10)

        self.sub_robot_state = self.create_subscription(
            String, '/robot_internal_state', self.state_callback, 10
        )
        self.robot_state = None
        self.state_override = False

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

        self.rgb_sub = message_filters.Subscriber(self, Image, "/top_camera/rgb/preview/image_raw")
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, "/top_camera/rgb/preview/depth/points")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.pc_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.groups = []
        self.group_threshold = 0.5
        self.min_detections = 3

        self.get_logger().info("🦜 Bird detection node ready (mode-split active)")

    def state_callback(self, msg):
        self.robot_state = msg.data

    def synced_callback(self, img_msg, pc_msg):
        if self.robot_state is None and not self.state_override:
            return

        if self.robot_state == "ARRIVED_AT_GOAL" or self.state_override:
            self.handle_arrived_at_goal(img_msg, pc_msg)
        elif self.robot_state == "ARRIVED_AT_BIRD":
            self.handle_moving_closer_to_bird(img_msg, pc_msg)
        # else: do nothing

    def handle_arrived_at_goal(self, img_msg, pc_msg):
        """Only YOLO detection, publish PointStamped for first bird found."""
        img_bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        results = self.yolo_model.predict(source=[img_bgr], conf=0.7, save=False, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            u_center, v_center = (x1 + x2) // 2, (y1 + y2) // 2
            u_range = slice(max(u_center - 2, 0), u_center + 3)
            v_range = slice(max(v_center - 2, 0), v_center + 3)

            pc_array = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
            try:
                points_reshaped = pc_array.reshape((pc_msg.height, pc_msg.width, 3))
            except Exception as e:
                self.get_logger().warn(f"Failed to reshape point cloud: {e}")
                continue

            points_sample = points_reshaped[v_range, u_range].reshape(-1, 3)
            valid_points = points_sample[np.isfinite(points_sample).all(axis=1) & (np.linalg.norm(points_sample, axis=1) > 0.05)]

            if valid_points.shape[0] < 3:
                self.get_logger().warn("Not enough valid points for initial detection.")
                continue

            avg_3d = np.mean(valid_points, axis=0)
            stamped_point = PointStamped()
            stamped_point.header.stamp = self.get_clock().now().to_msg()
            stamped_point.header.frame_id = img_msg.header.frame_id
            stamped_point.point.x = float(avg_3d[0])
            stamped_point.point.y = float(avg_3d[1])
            stamped_point.point.z = float(avg_3d[2])

            try:
                transform = self.tf_buffer.lookup_transform('map', img_msg.header.frame_id, rclpy.time.Time())
                map_point_stamped = tf2_geometry_msgs.do_transform_point(stamped_point, transform)
                self.init_bird_pub.publish(map_point_stamped)
                self.get_logger().info(f"🟣 Initial bird position published at {np.round([map_point_stamped.point.x, map_point_stamped.point.y, map_point_stamped.point.z], 2)}")
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue
            break  # Only publish first bird
        # No need to call group logic

    def handle_moving_closer_to_bird(self, img_msg, pc_msg):
        """YOLO + classification + grouping + DetectedBird publishing, with visualization."""
        img_bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        vis_img = img_bgr.copy()  # for drawing
        results = self.yolo_model.predict(source=[img_bgr], conf=0.7, save=False, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Draw bounding box on full image
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = self.transform(transforms.ToPILImage()(crop_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.resnet(crop_tensor)
                probabilities = torch.softmax(pred, dim=1)
                class_confidence, class_idx = torch.max(probabilities, dim=1)
                class_confidence = class_confidence.item()

                if class_confidence < 0.4:
                    self.get_logger().info(f"Skipped classification {self.class_names[class_idx.item()]} (low confidence: {class_confidence:.2f})")
                    continue
                
                class_name = self.class_names[class_idx.item()]
                self.get_logger().info(f"Detected Bird: {class_name} Confidence: {class_confidence}")

            # Draw label on full image
            label = f"{class_name}: {class_confidence:.2f}"
            cv2.putText(vis_img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # ----------- VISUALIZE FULL IMAGE AND CROPPED IMAGE ------------
            cv2.imshow("Bird Detector - RGB", vis_img)
            cv2.waitKey(1)
            # ----------------------------------------------------------------

            pc_array = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
            try:
                points_reshaped = pc_array.reshape((pc_msg.height, pc_msg.width, 3))
            except Exception as e:
                self.get_logger().warn(f"Failed to reshape point cloud: {e}")
                continue

            points_sample = points_reshaped[y1:y2, x1:x2].reshape(-1, 3)
            valid_points = points_sample[np.isfinite(points_sample).all(axis=1)]

            if valid_points.shape[0] < 3:
                self.get_logger().warn("Not enough valid points.")
                continue

            avg_3d = np.median(valid_points, axis=0)
            stamped_point = PointStamped()
            stamped_point.header.stamp = self.get_clock().now().to_msg()
            stamped_point.header.frame_id = img_msg.header.frame_id
            stamped_point.point.x = float(avg_3d[0])
            stamped_point.point.y = float(avg_3d[1])
            stamped_point.point.z = float(avg_3d[2])

            try:
                transform = self.tf_buffer.lookup_transform('map', img_msg.header.frame_id, rclpy.time.Time())
                map_point = tf2_geometry_msgs.do_transform_point(stamped_point, transform).point
                robot_position = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                bird_position = np.array([map_point.x, map_point.y, map_point.z])
                distance = np.linalg.norm(bird_position - robot_position)

                if distance > 1.0:
                    self.get_logger().info(f"🛑 Skipping bird (too far: {distance:.2f}m)")
                    continue

                self.add_to_group(map_point, class_name, class_confidence)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")

        self.publish_groups()


    def add_to_group(self, point, class_name, confidence):
        position = np.array([point.x, point.y, point.z])
        for group in self.groups:
            if np.linalg.norm(group['position'] - position) < self.group_threshold:
                group['positions'].append(position)
                group['classifications'].append((class_name, confidence))
                return

        self.groups.append({
            'positions': [position],
            'classifications': [(class_name, confidence)],
            'position': position
        })

    def publish_groups(self):
        for group in self.groups[:]:  # Safe removal
            if len(group['positions']) >= self.min_detections:
                avg_pos = np.mean(group['positions'], axis=0)

                # Aggregate classification confidences
                class_scores = {}
                total_weight = 0.0

                for cls, conf in group['classifications']:
                    if cls not in class_scores:
                        class_scores[cls] = 0.0
                    class_scores[cls] += conf
                    total_weight += conf

                if total_weight == 0:
                    continue

                # Normalize and choose best class
                normalized_scores = {cls: score / total_weight for cls, score in class_scores.items()}
                final_class = max(normalized_scores.items(), key=lambda x: x[1])[0]
                final_confidence = normalized_scores[final_class]

                msg = DetectedBird()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.position = Point(x=float(avg_pos[0]), y=float(avg_pos[1]), z=float(avg_pos[2]))
                msg.class_name = final_class
                msg.confidence = float(final_confidence)

                self.bird_pub.publish(msg)
                self.get_logger().info(f"🟢 Published bird: {final_class}, Weighted Confidence: {final_confidence:.2f}, Pos: {avg_pos.round(2)}")

                self.groups.remove(group)

def main(args=None):
    rclpy.init(args=args)
    node = BirdDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
