#!/usr/bin/env python3

import rclpy
import math
import random
import numpy as np
import cv2
from PIL import Image as PILImage

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Vector3
from std_msgs.msg import Header
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO

# Gender classification
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch

from dis_tutorial3.msg import DetectedFace  # Custom message

class FaceDetector(Node):
    def __init__(self):
        super().__init__('detect_people')
        self.device = self.declare_parameter('device', '').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.faces = []  # Now stores (cx, cy, gender, confidence)
        self.face_groups = []
        self.detected_faces_sent = set()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # YOLO model for face detection
        self.model = YOLO("yolov8n.pt")
        
        # Gender classification model
        self.get_logger().info("Loading gender classification model...")
        try:
            self.gender_processor = AutoImageProcessor.from_pretrained("dima806/fairface_gender_image_detection")
            self.gender_model = AutoModelForImageClassification.from_pretrained("dima806/fairface_gender_image_detection")
            
            # Set device for gender model
            if self.device and torch.cuda.is_available():
                self.gender_model = self.gender_model.to(self.device)
            
            self.get_logger().info("✅ Gender classification model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load gender model: {e}")
            self.gender_model = None
            self.gender_processor = None

        self.sub_rgb = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.sub_pc = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pc_callback, qos_profile_sensor_data)

        self.sub_robot_state = self.create_subscription(
            String,
            '/robot_internal_state',
            self.state_callback,
            10
        )
        self.robot_state = None
        self.state_override = False

        self.face_pub = self.create_publisher(DetectedFace, "/detected_faces", 10)

        self.timer = self.create_timer(1.0, self.publish_new_faces)

        self.face_confidence_threshold = 0.6
        self.face_depth_check = 1.5
        self.number_of_detections_threshold = 5

        self.get_logger().info("✅ detect_people running. Waiting for faces...")

    def state_callback(self, msg):
        self.robot_state = msg.data


    def rgb_callback(self, msg):
        self.faces.clear()

        if (self.robot_state is None or (self.robot_state != "MOVING_TO_GOAL" and self.robot_state != "SELECTING_NEW_GOAL")) and self.state_override is False:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model.predict(img, imgsz=(256, 320), conf = self.face_confidence_threshold, show=False, verbose=False, classes=[0], device=self.device)

            # Create a copy of the image for display
            display_img = img.copy()
            
            for r in results:
                for bbox in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, bbox)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Extract face region for gender classification
                    face_region = img[y1:y2, x1:x2]
                    
                    # Classify gender if face region is large enough
                    if face_region.shape[0] > 30 and face_region.shape[1] > 30:
                        gender, confidence = self.classify_gender(face_region)
                    else:
                        gender, confidence = "unknown", 0.0
                    
                    self.faces.append((cx, cy, gender, confidence))
                    
                    # Choose color based on gender: blue for men, pink for women, green for unknown
                    if gender == "man":
                        color = (255, 0, 0)  # Blue in BGR
                        label_text = f"Man ({confidence:.2f})"
                    elif gender == "woman":
                        color = (255, 0, 255)  # Pink/Magenta in BGR
                        label_text = f"Woman ({confidence:.2f})"
                    else:
                        color = (0, 255, 0)  # Green for unknown
                        label_text = f"Unknown"
                    
                    # Draw bounding box around detected face
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    cv2.putText(display_img, label_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw center point
                    cv2.circle(display_img, (cx, cy), 3, (0, 0, 255), -1)

            # Add status text
            status_text = f"Faces detected: {len(self.faces)}"
            cv2.putText(display_img, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the image
            cv2.imshow('Face Detection - Press Q to quit', display_img)
            
            # Handle window events (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit key pressed. Shutting down...")
                rclpy.shutdown()

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

        for face_data in self.faces:
            cx, cy, gender, confidence = face_data
            self.get_logger().warn(f"Face found at ({cx},{cy}) - {gender} ({confidence:.2f})")

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
                    map_normal,
                    gender,
                    confidence
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

    
    def add_to_group(self, new_point, normal, gender, confidence, threshold=0.5):
        for group in self.face_groups:
            if np.linalg.norm(group['point'] - new_point) < threshold:
                group['points'].append(new_point)
                group['normals'].append(normal)
                group['genders'].append(gender)
                group['confidences'].append(confidence)
                return
        self.face_groups.append({'points': [new_point], 'normals': [normal], 'genders': [gender], 'confidences': [confidence], 'point': new_point})

    def publish_new_faces(self):
        for i, group in enumerate(self.face_groups):
            if len(group['points']) < self.number_of_detections_threshold:
                #self.get_logger().warn(f"too little measurements. {i}")
                continue  # Not enough observations for reliable estimate

            avg_pos = np.mean(group['points'], axis=0)
            avg_norm = np.mean(group['normals'], axis=0)
            
            # Determine most common gender
            genders = group['genders']
            confidences = group['confidences']
            
            # Find the most confident gender prediction
            if genders and confidences:
                max_conf_idx = np.argmax(confidences)
                best_gender = genders[max_conf_idx]
                best_confidence = confidences[max_conf_idx]
            else:
                best_gender = "unknown"
                best_confidence = 0.0
            
            if best_gender == "unknown":
                continue
            
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

            msg.gender = best_gender

            self.face_pub.publish(msg)
            self.detected_faces_sent.add(key)

            self.get_logger().info(
                f"🧍 Published reliable face (n={len(group['points'])}) at {avg_pos}, normal: {avg_norm}, gender: {best_gender} ({best_confidence:.2f})"
            )
    
    def destroy_node(self):
        """Clean up OpenCV windows when node is destroyed"""
        cv2.destroyAllWindows()
        super().destroy_node()

    def classify_gender(self, face_img):
        """Classify gender of a face image"""
        if self.gender_model is None or self.gender_processor is None:
            return "unknown", 0.0
        
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(face_rgb)
            
            # Process the image
            inputs = self.gender_processor(pil_image, return_tensors="pt")
            
            if self.device and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get the predicted class and confidence
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
            
            # Map class ID to gender (this may need adjustment based on the model's labels)
            # For fairface model: typically 0=Female, 1=Male
            gender = "woman" if predicted_class_id == 0 else "man"
            
            return gender, confidence
            
        except Exception as e:
            self.get_logger().warn(f"Gender classification failed: {e}")
            return "unknown", 0.0

    def get_latest_face_genders(self):
        """Get the latest detected faces with their gender classifications"""
        face_genders = []
        for face_data in self.faces:
            cx, cy, gender, confidence = face_data
            face_genders.append({
                'position': (cx, cy),
                'gender': gender,
                'confidence': confidence
            })
        return face_genders
    
    def get_most_confident_gender(self):
        """Get the gender of the most confidently detected face"""
        if not self.faces:
            return None, 0.0
        
        max_confidence = 0.0
        best_gender = None
        
        for face_data in self.faces:
            cx, cy, gender, confidence = face_data
            if confidence > max_confidence and gender != "unknown":
                max_confidence = confidence
                best_gender = gender
        
        return best_gender, max_confidence

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()