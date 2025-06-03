#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import numpy as np
import cv2
import math
import transforms3d.euler
import os
from ament_index_python.packages import get_package_share_directory


class Planner(Node):
    def __init__(self):
        super().__init__('inspection_marker_publisher')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_markers = self.create_publisher(MarkerArray, '/inspection_markers', qos)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)

        self.map_received = False
        self.map_img = None
        self.dist_transform = None
        self.resolution = None

        # Thresholds from map.yaml
        self.occupied_thresh = 0.90
        self.free_thresh = 0.25

        # Marker generation settings
        self.cam_offset = 0.7
        self.target_offset = 0.2
        self.spacing = 0.3
        self.max_line_length_m = 2.0
        self.min_clearance_m = 0.5

        self.max_push_m = 1.0    # maximum distance to push away from wall
        self.push_step_m = 0.05  # step size when pushing

    def map_callback(self, msg):
        if self.map_received:
            return
        self.map_received = True

        self.resolution = msg.info.resolution
        origin = msg.info.origin.position
        width = msg.info.width
        height = msg.info.height

        # Load and flip PGM map (origin is at bottom left)
        package_path = get_package_share_directory('dis_tutorial3')
        map_path = os.path.join(package_path, 'maps', 'map.pgm')
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error("❌ Failed to load map.pgm")
            return
        self.map_img = cv2.flip(img, 0)

        # Generate occupancy map
        occ = np.full_like(self.map_img, -1.0, dtype=np.float32)
        occ[self.map_img >= int(self.free_thresh * 255)] = 1.0
        occ[self.map_img <= int(self.occupied_thresh * 255)] = 0.0

        # Compute distance transform (for clearance)
        binary = (occ == 1.0).astype(np.uint8)
        self.dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Detect walls and lines
        edges = cv2.Canny((occ == 0).astype(np.uint8) * 255, 50, 150)
        lines = cv2.HoughLinesP(edges, 0.1, np.pi / 180, threshold=5, minLineLength=5, maxLineGap=5)
        if lines is None:
            self.get_logger().warn("No walls detected.")
            return

        all_lines = []
        for l in lines:
            all_lines.extend(self.split_line(*l[0], self.max_line_length_m, self.resolution))

        poses, _ = self.generate_camera_targets(all_lines, occ, self.resolution, origin, height, start_target_id=1000)
        markers = self.generate_markers(poses)
        self.pub_markers.publish(markers)
        self.get_logger().info(f"✅ Published {len(markers.markers)} markers")

    def split_line(self, x1, y1, x2, y2, max_len, res):
        max_len_px = max_len / res
        dist = np.hypot(x2 - x1, y2 - y1)
        num = max(1, int(np.ceil(dist / max_len_px)))
        return [[[int(x1 + (x2 - x1) * i / num), int(y1 + (y2 - y1) * i / num),
                  int(x1 + (x2 - x1) * (i + 1) / num), int(y1 + (y2 - y1) * (i + 1) / num)]]
                for i in range(num)]

    def pixel_to_world(self, x_pix, y_pix, res, origin, height):
        x = origin.x + x_pix * res
        y = origin.y + y_pix * res
        return np.array([x, y])

    def world_to_pixel(self, wx, wy, origin, res):
        px = int((wx - origin.x) / res)
        py = int((wy - origin.y) / res)
        return px, py

    def is_valid(self, x, y, grid, buf=1):
        x, y = int(round(x)), int(round(y))
        if x < buf or y < buf or x >= grid.shape[1] - buf or y >= grid.shape[0] - buf:
            return False
        region = grid[y - buf:y + buf + 1, x - buf:x + buf + 1]
        return np.all(region == 1.0)

    def has_clearance(self, wx, wy, origin, min_clearance=None):
        if self.dist_transform is None:
            return True
        px, py = self.world_to_pixel(wx, wy, origin, self.resolution)
        if 0 <= px < self.dist_transform.shape[1] and 0 <= py < self.dist_transform.shape[0]:
            clearance = self.dist_transform[py, px] * self.resolution
            if min_clearance is None:
                min_clearance = self.min_clearance_m
            return clearance >= min_clearance
        return False

    def push_away_from_closest_wall(self, wx, wy, origin, occ, res, min_clearance, max_push=None, step=None):
        """
        Move point (wx, wy) directly away from the closest wall pixel until min_clearance is reached or max_push is exceeded.
        Returns (new_wx, new_wy) or None if not found.
        """
        if max_push is None:
            max_push = self.max_push_m
        if step is None:
            step = self.push_step_m

        candidate = np.array([wx, wy])
        total_push = 0.0
        for _ in range(int(max_push // step)):
            if self.has_clearance(candidate[0], candidate[1], origin, min_clearance):
                return candidate[0], candidate[1]
            wall_px = self.closest_wall_pixel(candidate[0], candidate[1], origin, res, occ)
            if wall_px is None:
                return candidate[0], candidate[1]
            # Convert wall pixel to world coordinates
            wall_w = self.pixel_to_world(wall_px[0], wall_px[1], res, origin, self.map_img.shape[0])
            away_vec = candidate - wall_w
            norm = np.linalg.norm(away_vec)
            if norm < 1e-3:
                # If we're right on top of a wall pixel, nudge randomly
                direction = np.array([1.0, 0.0])
            else:
                direction = away_vec / norm
            candidate = candidate + direction * step
            total_push += step
            if total_push > max_push:
                break
        return None


    def closest_wall_pixel(self, wx, wy, origin, res, occ):
        # Convert world to pixel
        px, py = self.world_to_pixel(wx, wy, origin, res)
        wall_pixels = np.argwhere(occ == 0.0)
        if wall_pixels.shape[0] == 0:
            return None
        # Compute squared distances for efficiency
        dists = (wall_pixels[:, 1] - px) ** 2 + (wall_pixels[:, 0] - py) ** 2
        min_idx = np.argmin(dists)
        closest = wall_pixels[min_idx]
        # Return pixel coordinates (x, y)
        return int(closest[1]), int(closest[0])


    def push_away_from_wall(self, wx, wy, norm_vec, origin, min_clearance, max_push=None, step=None):
        """
        Move point (wx, wy) along norm_vec until min_clearance is reached or max_push is exceeded.
        Returns (new_wx, new_wy) or None if not found.
        """
        if max_push is None:
            max_push = self.max_push_m
        if step is None:
            step = self.push_step_m
        for d in np.arange(0, max_push + step, step):
            candidate = np.array([wx, wy]) + norm_vec * d
            if self.has_clearance(candidate[0], candidate[1], origin, min_clearance):
                return candidate[0], candidate[1]
        return None

    def generate_camera_targets(self, lines, grid, res, origin, height, start_target_id=1000):
        spacing_px = self.spacing / res
        cam_offset_px = self.cam_offset / res
        target_offset_px = self.target_offset / res
        poses = []
        target_id = start_target_id

        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_pix = np.array([(x1 + x2)/2, (y1 + y2)/2])
            mid_world = self.pixel_to_world(mid_pix[0], mid_pix[1], res, origin, height)
            dir_vec = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            if np.linalg.norm(dir_vec) == 0:
                continue
            dir_vec /= np.linalg.norm(dir_vec)
            norm_vec = np.array([-dir_vec[1], dir_vec[0]])
            num_pts = max(1, int(np.linalg.norm([x2 - x1, y2 - y1]) / spacing_px))

            for direction in [1, -1]:
                targets = []
                for i in range(num_pts + 1):
                    alpha = i / num_pts
                    interp = np.array([x1, y1]) * (1 - alpha) + np.array([x2, y2]) * alpha
                    offset = interp + direction * target_offset_px * norm_vec
                    if self.is_valid(offset[0], offset[1], grid):
                        wp = self.pixel_to_world(offset[0], offset[1], res, origin, height)
                        # Optionally push targets away too:
                        # if not self.has_clearance(wp[0], wp[1], origin):
                        #     pushed = self.push_away_from_wall(wp[0], wp[1], norm_vec * direction, origin, self.min_clearance_m)
                        #     if pushed is not None:
                        #         wp = np.array(pushed)
                        #     else:
                        #         continue
                        targets.append((wp[0], wp[1], norm_vec[0] * direction, norm_vec[1] * direction, target_id))
                        target_id += 1

                if not targets:
                    continue

                center_offset = mid_pix + direction * cam_offset_px * norm_vec
                wp = self.pixel_to_world(center_offset[0], center_offset[1], res, origin, height)
                yaw = math.atan2(mid_world[1] - wp[1], mid_world[0] - wp[0])

                if self.is_valid(center_offset[0], center_offset[1], grid):
                    # If not enough clearance, push away from wall
                    if not self.has_clearance(wp[0], wp[1], origin):
                        pushed = self.push_away_from_closest_wall(wp[0], wp[1], origin, grid, res, self.min_clearance_m)

                        if pushed is not None:
                            wp = np.array(pushed)
                        else:
                            continue  # Could not push far enough from wall
                    poses.append({'pose': (wp[0], wp[1], yaw), 'targets': targets})

        return poses, target_id

    def generate_markers(self, cam_targets):
        markers = MarkerArray()
        cam_marker_id = 0

        for entry in cam_targets:
            x, y, yaw = entry['pose']
            q = transforms3d.euler.euler2quat(0, 0, yaw, axes='sxyz')
            q = (q[1], q[2], q[3], q[0])

            cam = Marker()
            cam.header.frame_id = "map"
            cam.ns = "inspection"
            cam.id = cam_marker_id
            cam_marker_id += 1
            cam.type = Marker.ARROW
            cam.action = Marker.ADD
            cam.pose.position.x = x
            cam.pose.position.y = y
            cam.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            cam.scale.x = 0.5
            cam.scale.y = 0.1
            cam.scale.z = 0.1
            cam.color.r = 0.0
            cam.color.g = 0.0
            cam.color.b = 1.0
            cam.color.a = 1.0
            markers.markers.append(cam)

        return markers


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()