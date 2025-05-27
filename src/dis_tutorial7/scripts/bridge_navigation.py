#!/usr/bin/env python3

import rclpy
import numpy as np
import cv2
import math
import random

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, String
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from tf_transformations import quaternion_from_euler

import tf2_ros
import tf2_geometry_msgs


class BridgeNavigator(Node):
    def __init__(self):
        super().__init__('bride_movement')
        self.device = self.declare_parameter('device', '').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.sub_rgb = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.sub_arm_cam = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.arm_rgb_callback, qos_profile_sensor_data)
        self.sub_arm_depth = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.arm_depth_callback, qos_profile_sensor_data)
        self.sub_arm_pointcloud = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.pc_callback, qos_profile_sensor_data)

        self.pub_bridge_pose = self.create_publisher(PoseStamped, "/bridge_pose_map", 10)
        self.latest_pointcloud = None

        #self.arm_command_pub = self.create_publisher(String, "/arm_command", 10)
        #self.initial_pose_timer = self.create_timer(3, self.publish_initial_command)

        self.display_dict = {
            "Original (Arm RGB)": np.zeros((240, 320, 3), dtype=np.uint8),
            "Hue Channel": np.zeros((240, 320), dtype=np.uint8),
            "Binary (Otsu on Hue)": np.zeros((240, 320), dtype=np.uint8),
            "After Closing": np.zeros((240, 320), dtype=np.uint8),
            "Canny Edges": np.zeros((240, 320), dtype=np.uint8),
            "Fitted Guardrails": np.zeros((240, 320, 3), dtype=np.uint8)
        }

        self.get_logger().info("Bridge mover started")

    def pc_callback(self, msg):
        self.latest_pointcloud = msg

    def publish_initial_command(self):
        msg = String()
        base_link_bend = 0.45
        bend_factor = 0.6
        yaw = 0.0
        link1_rotation = base_link_bend
        link2_rotation = bend_factor - base_link_bend
        link3_rotation = np.pi - bend_factor - base_link_bend
        msg.data = f"manual:[{yaw},{link1_rotation},{link2_rotation},{link3_rotation}]"
        self.arm_command_pub.publish(msg)
        self.get_logger().info("Published initial arm command: look_for_parking")
        self.initial_pose_timer.cancel()

    def label_image(self, img, label_text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_width, text_height = text_size

        labeled_img = np.zeros((img.shape[0] + text_height + 10, img.shape[1], 3), dtype=np.uint8)
        labeled_img[text_height + 10:, :, :] = img

        text_x = (img.shape[1] - text_width) // 2
        text_y = text_height + 2
        cv2.putText(labeled_img, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return labeled_img

    def add_padding(self, img, pad=10, color=(255, 255, 255)):
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)

    def display_image_grid(self, image_dict, window_name="Ring Detection Overview", rows=2):
        labeled_images = []

        for label, img in image_dict.items():
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            labeled = self.label_image(img, label)
            padded = self.add_padding(labeled)
            labeled_images.append(padded)

        columns = []
        for i in range(0, len(labeled_images), rows):
            col_imgs = labeled_images[i:i+rows]
            if len(col_imgs) < rows:
                h, w = col_imgs[0].shape[:2]
                white = np.ones((h, w, 3), dtype=np.uint8) * 255
                col_imgs += [white] * (rows - len(col_imgs))
            columns.append(cv2.vconcat(col_imgs))

        grid = cv2.hconcat(columns)
        cv2.imshow(window_name, grid)
        cv2.waitKey(1)

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

    def arm_rgb_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            blurred = cv2.GaussianBlur(hue, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)

            SE_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            pad = 40
            binary_padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            closed_padded = cv2.morphologyEx(binary_padded, cv2.MORPH_CLOSE, SE_closing)
            closed = closed_padded[pad:-pad, pad:-pad]

            edges = cv2.Canny(closed, 50, 150)

            height, width = closed.shape
            line_vis = img_bgr.copy()

            self.display_dict["Original (Arm RGB)"] = img_bgr
            self.display_dict["Hue Channel"] = hue
            self.display_dict["Binary (Otsu on Hue)"] = binary
            self.display_dict["After Closing"] = closed
            self.display_dict["Canny Edges"] = edges

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
            if (num_labels - 1 < 2):
                self.display_image_grid(self.display_dict)
                return

            min_pixels = 40
            segment_size = 60
            all_lines = []

            for label in range(1, num_labels):
                mask = (labels == label).astype(np.uint8)
                ys, xs = np.nonzero(mask)
                if len(xs) < min_pixels:
                    continue

                points = np.vstack((xs, ys)).T.astype(np.float32).reshape(-1, 1, 2)
                sorted_indices = np.argsort(points[:, 0, 1])
                sorted_points = points[sorted_indices][:, 0, :]

                num_segments = len(sorted_points) // segment_size
                for i in range(num_segments):
                    segment = sorted_points[i * segment_size: (i + 1) * segment_size]
                    if len(segment) < 2:
                        continue

                    segment_pts = segment.astype(np.float32).reshape(-1, 1, 2)
                    [vx, vy, x0, y0] = cv2.fitLine(segment_pts, cv2.DIST_L2, 0, 0.01, 0.01)

                    pt1 = tuple(segment[0].astype(int))
                    pt2 = tuple(segment[-1].astype(int))
                    center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    angle_rad = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                    angle_deg = math.degrees(angle_rad)

                    color = tuple(random.randint(50, 255) for _ in range(3))
                    cv2.line(line_vis, pt1, pt2, color, 1)

                    all_lines.append({
                        "pt1": pt1,
                        "pt2": pt2,
                        "center": center,
                        "angle": angle_deg,
                        "color": color,
                        "edge_label": label
                    })

            if all_lines:
                image_center_x = width // 2
                angle_thresh_deg = 60.0
                min_x_separation = width * 0.2
                max_x_separation = width * 0.8
                number_of_pairs_generated = 5

                left_lines = [l for l in all_lines if l["center"][0] < image_center_x]
                right_lines = [l for l in all_lines if l["center"][0] >= image_center_x]

                left_lines = sorted(left_lines, key=lambda l: -l["center"][1])
                right_lines = sorted(right_lines, key=lambda l: -l["center"][1])

                
                valid_pairs = []

                used_left = set()
                used_right = set()

                for l in left_lines:
                    if len(valid_pairs) >= number_of_pairs_generated:
                        break
                    for r in right_lines:
                        if len(valid_pairs) >= number_of_pairs_generated:
                            break
                        if l['edge_label'] == r['edge_label']:
                            continue
                        angle_diff = abs(l["angle"] - r["angle"])
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        if angle_diff > angle_thresh_deg:
                            continue
                        x_diff = abs(l["center"][0] - r["center"][0])
                        if not (min_x_separation <= x_diff <= max_x_separation):
                            continue
                        if id(l) in used_left or id(r) in used_right:
                            continue
                        valid_pairs.append((l, r))
                        used_left.add(id(l))
                        used_right.add(id(r))

                for idx, (l, r) in enumerate(valid_pairs):
                    pair_color = tuple(random.randint(100, 255) for _ in range(3))
                    cv2.line(line_vis, l["pt1"], l["pt2"], pair_color, 3)
                    cv2.line(line_vis, r["pt1"], r["pt2"], pair_color, 3)

                    midpoint = np.array([
                        (l["center"][0] + r["center"][0]) // 2,
                        (l["center"][1] + r["center"][1]) // 2
                    ], dtype=np.int32)

                    l_vec = np.array([l["pt2"][0] - l["pt1"][0], l["pt2"][1] - l["pt1"][1]], dtype=np.float32)
                    r_vec = np.array([r["pt2"][0] - r["pt1"][0], r["pt2"][1] - r["pt1"][1]], dtype=np.float32)

                    l_dir = l_vec / (np.linalg.norm(l_vec) + 1e-6)
                    r_dir = r_vec / (np.linalg.norm(r_vec) + 1e-6)
                    avg_dir = (l_dir + r_dir) / 2.0
                    avg_dir /= (np.linalg.norm(avg_dir) + 1e-6)

                    arrow_length = 50
                    tip = (midpoint + (avg_dir * -arrow_length)).astype(int)
                    cv2.arrowedLine(line_vis, tuple(midpoint), tuple(tip), (255, 255, 0), 3, tipLength=0.2)
                    cv2.circle(line_vis, tuple(midpoint), 5, (255, 0, 255), -1)

                    if idx == len(valid_pairs) - 1:
                        if self.latest_pointcloud is None:
                            continue
                        try:
                            pc_array = pc2.read_points_numpy(self.latest_pointcloud, field_names=("x", "y", "z"))
                            pc_array = pc_array.reshape((self.latest_pointcloud.height, self.latest_pointcloud.width, 3))

                            pt = pc_array[midpoint[1], midpoint[0]]
                            if not np.all(np.isfinite(pt)) or np.linalg.norm(pt) < 0.05:
                                continue

                            camera_point = PointStamped()
                            camera_point.header.stamp = self.get_clock().now().to_msg()
                            camera_point.header.frame_id = self.latest_pointcloud.header.frame_id
                            camera_point.point.x = float(pt[0])
                            camera_point.point.y = float(pt[1])
                            camera_point.point.z = float(pt[2])

                            transform = self.tf_buffer.lookup_transform(
                                "map",
                                camera_point.header.frame_id,
                                rclpy.time.Time(),
                                timeout=rclpy.duration.Duration(seconds=0.5)
                            )
                            map_point = tf2_geometry_msgs.do_transform_point(camera_point, transform)

                            offset = math.radians(90)
                            yaw = math.atan2(-avg_dir[1], -avg_dir[0]) + offset
                            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

                            pose_msg = PoseStamped()
                            pose_msg.header = map_point.header
                            pose_msg.pose.position = map_point.point
                            pose_msg.pose.orientation.x = qx
                            pose_msg.pose.orientation.y = qy
                            pose_msg.pose.orientation.z = qz
                            pose_msg.pose.orientation.w = qw

                            self.pub_bridge_pose.publish(pose_msg)
                        except Exception as e:
                            self.get_logger().warn(f"Exception during bridge detection: {e}")

            self.display_dict["Fitted Guardrails"] = line_vis
            self.display_image_grid(self.display_dict)

        except Exception as e:
            self.get_logger().warn(f"Exception during bridge detection: {e}")

    def arm_depth_callback(self, msg):
        return


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()