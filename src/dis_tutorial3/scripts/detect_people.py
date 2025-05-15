#!/usr/bin/env python3

import rclpy
import math
import random
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Vector3
from std_msgs.msg import Header
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3Stamped

import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO

from dis_tutorial3.msg import DetectedFace  # Custom message

class FaceDetector(Node):
    def __init__(self):
        super().__init__('detect_people')
        self.device = self.declare_parameter('device', '').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.faces = []
        self.face_groups = []
        self.detected_faces_sent = set()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.model = YOLO("yolov8n.pt")

        self.sub_rgb = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.sub_pc = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pc_callback, qos_profile_sensor_data)

        self.face_pub = self.create_publisher(DetectedFace, "/detected_faces", 10)

        self.timer = self.create_timer(1.0, self.publish_new_faces)

        self.face_confidence_threshold = 0.6
        self.face_depth_check = 1.5
        self.number_of_detections_threshold = 5

        self.get_logger().info("✅ detect_people running. Waiting for faces...")

    def rgb_callback(self, msg):
        self.faces.clear()

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model.predict(img, imgsz=(256, 320), conf = self.face_confidence_threshold, show=False, verbose=False, classes=[0], device=self.device)

            for r in results:
                for bbox in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, bbox)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    self.faces.append((cx, cy))

        except Exception as e:
            self.get_logger().warn(f"Failed to process image: {e}")


    def pc_callback(self, msg):
        if not self.faces:
            return

        try:
            pc_array = pc2.read_points_numpy(msg, field_names=("x", "y", "z")).reshape((msg.height, msg.width, 3))
        except Exception as e:
            self.get_logger().warn(f"Failed to parse point cloud: {e}")
            return

        for cx, cy in self.faces:
            self.get_logger().warn(f"Face found at ({cx},{cy})")

            pc_check_depth = pc_array[cy, cx, :]
            if np.isnan(pc_check_depth).any() or np.linalg.norm(pc_check_depth) > self.face_depth_check:
                self.get_logger().warn("Face is too far or invalid depth.")
                continue

            window = 10
            region = pc_array[max(0, cy - window):cy + window, max(0, cx - window):cx + window, :]
            points = region.reshape(-1, 3)
            points = points[~np.isnan(points).any(axis=1)]

            if len(points) < 30:
                continue

            normal, centroid = self.fit_plane(points)
            if normal is None or not np.all(np.isfinite(centroid)) or not np.all(np.isfinite(normal)):
                self.get_logger().warn("Plane fitting failed or returned invalid values.")
                continue

            if np.linalg.norm(normal) < 1e-3:
                self.get_logger().warn("Normal vector too small, skipping.")
                continue

            # Flip normal to face the camera
            camera_origin = np.array([0.0, 0.0, 0.0])
            to_centroid = centroid - camera_origin
            if np.dot(normal, to_centroid) > 0:
                normal = -normal

            offset = centroid + normal * 0.5
            if not np.all(np.isfinite(offset)):
                self.get_logger().warn("Offset point contains NaNs or infs, skipping.")
                continue

            try:
                # Properly create PointStamped
                stamped = PointStamped()
                stamped.header.stamp = self.get_clock().now().to_msg()
                stamped.header.frame_id = msg.header.frame_id
                stamped.point.x = float(offset[0])
                stamped.point.y = float(offset[1])
                stamped.point.z = float(offset[2])

                transform = self.tf_buffer.lookup_transform(
                    target_frame="map",
                    source_frame=msg.header.frame_id,
                    time=rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )

                transformed = tf2_geometry_msgs.do_transform_point(stamped, transform)

                # Prepare and validate the normal
                normal_msg = Vector3Stamped()
                normal_msg.header.stamp = rclpy.time.Time().to_msg()
                normal_msg.header.frame_id = stamped.header.frame_id
                normal_msg.vector.x = float(normal[0])
                normal_msg.vector.y = float(normal[1])
                normal_msg.vector.z = float(normal[2])

                transformed_normal = self.tf_buffer.transform(
                    normal_msg,
                    target_frame="map",
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )

                map_normal = np.array([
                    transformed_normal.vector.x,
                    transformed_normal.vector.y,
                    transformed_normal.vector.z
                ])

                if not np.all(np.isfinite(map_normal)):
                    self.get_logger().warn("Transformed normal contains NaNs.")
                    continue

                self.add_to_group(
                    np.array([transformed.point.x, transformed.point.y, transformed.point.z]),
                    map_normal
                )

            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")


    def fit_plane(self, points, threshold=0.015, max_iters=100):
        best_inliers = []
        best_normal = None

        for _ in range(max_iters):
            try:
                sample = points[random.sample(range(len(points)), 3)]
                v1 = sample[1] - sample[0]
                v2 = sample[2] - sample[0]
                normal = np.cross(v1, v2)

                if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-3:
                    continue

                normal = normal / np.linalg.norm(normal)

                distances = np.abs(np.dot(points - sample[0], normal))
                inliers = points[distances < threshold]

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_normal = normal

            except Exception as e:
                self.get_logger().warn(f"Plane fitting iteration failed: {e}")
                continue

        if best_normal is not None and len(best_inliers) > 0:
            centroid = np.mean(best_inliers, axis=0)
            if not np.all(np.isfinite(centroid)):
                return None, None
            return best_normal, centroid

        return None, None

    
    def add_to_group(self, new_point, normal, threshold=0.5):
        for group in self.face_groups:
            if np.linalg.norm(group['point'] - new_point) < threshold:
                group['points'].append(new_point)
                group['normals'].append(normal)
                return
        self.face_groups.append({'points': [new_point], 'normals': [normal], 'point': new_point})

    def publish_new_faces(self):
        for i, group in enumerate(self.face_groups):
            if len(group['points']) < self.number_of_detections_threshold:
                #self.get_logger().warn(f"too little measurements. {i}")
                continue  # Not enough observations for reliable estimate

            avg_pos = np.mean(group['points'], axis=0)
            avg_norm = np.mean(group['normals'], axis=0)
            key = tuple(np.round(avg_pos, 2))

            if key in self.detected_faces_sent:
                continue

            msg = DetectedFace()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"

            msg.position.x = float(avg_pos[0])
            msg.position.y = float(avg_pos[1])
            msg.position.z = float(avg_pos[2])

            msg.normal.x = float(avg_norm[0])
            msg.normal.y = float(avg_norm[1])
            msg.normal.z = float(avg_norm[2])

            self.face_pub.publish(msg)
            self.detected_faces_sent.add(key)

            self.get_logger().info(
                f"🧍 Published reliable face (n={len(group['points'])}) at {avg_pos}, normal: {avg_norm}"
            )
    
def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

