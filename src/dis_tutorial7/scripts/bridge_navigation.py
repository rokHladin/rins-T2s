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
        #self.sub_pc = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pc_callback, qos_profile_sensor_data)
        self.sub_arm_cam = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.arm_rgb_callback, qos_profile_sensor_data)
        self.sub_arm_depth = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.arm_depth_callback, qos_profile_sensor_data)


        self.arm_command_pub = self.create_publisher(String, "/arm_command", 10)
        self.initial_pose_timer = self.create_timer(3, self.publish_initial_command)

        self.get_logger().info("Bridge mover started")


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
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            labeled = self.label_image(img, label)
            padded = self.add_padding(labeled)
            labeled_images.append(padded)

        # Create columns from images
        columns = []
        for i in range(0, len(labeled_images), rows):
            col_imgs = labeled_images[i:i+rows]
            # Make sure columns have equal length
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
            cv2.imshow("Front Camera", img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"Failed to process front camera image: {e}")


    def arm_rgb_callback(self, msg):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert BGR to HSV
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]  # Extract Hue channel

            # Apply Gaussian blur to hue
            blurred = cv2.GaussianBlur(hue, (5, 5), 0)

            # Apply Otsu's thresholding on hue
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)

            # Morphological operations to clean mask
            SE_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            pad = 40
            binary_padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad,
                                            borderType=cv2.BORDER_CONSTANT, value=0)
            closed_padded = cv2.morphologyEx(binary_padded, cv2.MORPH_CLOSE, SE_closing)
            closed = closed_padded[pad:-pad, pad:-pad]

            # Use Canny edges to get fine lines
            edges = cv2.Canny(closed, 50, 150)

            # Connected components on edges
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)

            height, width = closed.shape
            line_vis = img_bgr.copy()

            min_pixels = 40
            segment_size = 40

            all_lines = []

            
            if (num_labels - 1 < 2):
                self.get_logger().warn(f"Number of edge sections is not enough ({num_labels-1})")
                return

            # Loop through each label (skip label 0 = background)
            for label in range(1, num_labels):
                mask = (labels == label).astype(np.uint8)
                ys, xs = np.nonzero(mask)
                if len(xs) < min_pixels:
                    self.get_logger().warn(f"Edge section too small ({len(xs)} pixels)")
                    return

                points = np.vstack((xs, ys)).T.astype(np.float32).reshape(-1, 1, 2)
                sorted_indices = np.argsort(points[:, 0, 1])
                sorted_points = points[sorted_indices][:, 0, :]

                num_segments = len(sorted_points) // segment_size
                for i in range(num_segments):
                    segment = sorted_points[i * segment_size : (i + 1) * segment_size]
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
                        "edge_label" : label
                    })

            # Pair detection
            if all_lines:
                image_center_x = width // 2
                angle_thresh_deg = 45.0
                min_x_separation = width * 0.2

                # Separate into left and right groups
                left_lines = [l for l in all_lines if l["center"][0] < image_center_x]
                right_lines = [l for l in all_lines if l["center"][0] >= image_center_x]

                # Sort by Y descending (bottom-most first)
                left_lines = sorted(left_lines, key=lambda l: -l["center"][1])
                right_lines = sorted(right_lines, key=lambda l: -l["center"][1])

                best_pair = None

                # Try bottom-most pairs first
                for l in left_lines:
                    for r in right_lines:

                        if (l['edge_label'] == r['edge_label']):
                            continue

                        # Ensure they're roughly parallel
                        angle_diff = abs(l["angle"] - r["angle"])
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        if angle_diff > angle_thresh_deg:
                            continue

                        # Ensure they're far enough apart
                        x_diff = abs(l["center"][0] - r["center"][0])
                        if x_diff < min_x_separation:
                            continue

                        # Found a good-enough pair
                        best_pair = (l, r)
                        break  # take the first good pair
                    if best_pair:
                        break

                if best_pair:
                    l, r = best_pair
                    pair_color = tuple(random.randint(100, 255) for _ in range(3))
                    cv2.line(line_vis, l["pt1"], l["pt2"], pair_color, 3)
                    cv2.line(line_vis, r["pt1"], r["pt2"], pair_color, 3)

                    # Midpoint between guardrails
                    midpoint = np.array([
                        (l["center"][0] + r["center"][0]) // 2,
                        (l["center"][1] + r["center"][1]) // 2
                    ], dtype=np.int32)

                    # Compute direction vectors for both lines
                    l_vec = np.array([l["pt2"][0] - l["pt1"][0], l["pt2"][1] - l["pt1"][1]], dtype=np.float32)
                    r_vec = np.array([r["pt2"][0] - r["pt1"][0], r["pt2"][1] - r["pt1"][1]], dtype=np.float32)

                    # Normalize and average the direction
                    l_dir = l_vec / (np.linalg.norm(l_vec) + 1e-6)
                    r_dir = r_vec / (np.linalg.norm(r_vec) + 1e-6)
                    avg_dir = (l_dir + r_dir) / 2.0
                    avg_dir /= (np.linalg.norm(avg_dir) + 1e-6)  # Final normalization

                    # Draw direction vector (extend forward away from robot)
                    arrow_length = 50  # pixels
                    tip = (midpoint + (avg_dir * -arrow_length)).astype(int)  # negative to point away
                    cv2.arrowedLine(line_vis, tuple(midpoint), tuple(tip), (255, 255, 0), 3, tipLength=0.2)

                    # Draw midpoint
                    cv2.circle(line_vis, tuple(midpoint), 5, (255, 0, 255), -1)


            # Display all stages
            display_dict = {
                "Original (Arm RGB)": img_bgr,
                "Hue Channel": hue,
                "Binary (Otsu on Hue)": binary,
                "After Closing": closed,
                "Canny Edges": edges,
                "Fitted Guardrails": line_vis
            }

            self.display_image_grid(display_dict, window_name="Hue-based Water vs Bridge Masking")

        except Exception as e:
            self.get_logger().warn(f"Failed to process arm camera image: {e}")





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
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed on exit

if __name__ == '__main__':
    main()
