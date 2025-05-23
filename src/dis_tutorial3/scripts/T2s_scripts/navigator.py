#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
import transforms3d.euler
import heapq
from collections import deque
from geometry_msgs.msg import PointStamped

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from robot_commander import RobotCommander
from dis_tutorial3.msg import DetectedFace
from dis_tutorial3.msg import DetectedRing

from geometry_msgs.msg import PoseWithCovarianceStamped

import pyttsx3
from pyttsx3.engine import Engine

from enum import Enum, auto

class RobotState(Enum):
    INITIALIZING = auto()
    SELECTING_NEW_GOAL = auto()
    INSPECTING_GOAL = auto()
    SERVICE_FACE_DETECTION = auto()
    SERIVICE_RING_DETECTION = auto()
    ROBOT_FINISHED = auto()


class InspectionNavigator(Node):
    def __init__(self):
        super().__init__('navigator')

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.sub_markers = self.create_subscription(MarkerArray, '/inspection_markers', self.markers_callback, qos)

        self.pub_visited = self.create_publisher(MarkerArray, '/visited_inspection_markers', 10)

        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.pushed_face_pub = self.create_publisher(Marker, '/pushed_faces', 10)
        self.pub_ring_marker = self.create_publisher(MarkerArray, '/ring_markers', 10)

        self.create_subscription(
            DetectedFace,
            '/detected_faces',
            self.face_callback,
            qos_profile_sensor_data
        )


        self.create_subscription(
            DetectedRing,
            '/ring_position',
            self.ring_callback,
            qos_profile_sensor_data
        )


        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )
        
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)

        #initial robot state
        self.robot_state = RobotState.INITIALIZING

        #localization
        self.robot_pose = None
        self.odom_pose = None
        self.pose_sent = False
        self.retry_attempts = 0
        self.max_retries = 5
        self.retry_timer = self.create_timer(2.0, self.check_amcl_pose_timeout)

        #map info
        self.occupancy = None
        self.resolution = None
        self.origin = None

        #point visitation
        self.visiting_point_camera_poses = []
        self.current_visiting_map_point = None

        #face detection and ring detection
        self.face_queue = deque()
        self.ring_queue = deque()
        self.seen_faces = set()
        self.seen_rings = set()

        self.ring_color = None
        self.ring_visit_dist = 0.6
        
        #start up everything
        self.tts_engine = pyttsx3.init()
        self.cmdr = RobotCommander()
        self.timer = self.create_timer(1.0, self.robot_state_loop)


    def odom_callback(self, msg: Odometry):
        if self.pose_sent or self.robot_pose is not None:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])[2]

        self.odom_pose = (x, y, yaw)

        self.get_logger().info("📍 Using odometry to initialize AMCL pose")
        self.publish_initial_pose(x, y, math.degrees(yaw))
        self.pose_sent = True

    def set_initial_pose_once(self):
        if not self.pose_sent and self.robot_pose is None:
            self.publish_initial_pose(x=0.0, y=0.0, yaw_deg=0)
            self.pose_sent = True
            self.get_logger().info("📍 Published initial pose")

    def publish_initial_pose(self, x, y, yaw_deg):
        yaw_rad = math.radians(yaw_deg)
        q = transforms3d.euler.euler2quat(0, 0, yaw_rad, axes='sxyz')

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.x = q[1]
        msg.pose.pose.orientation.y = q[2]
        msg.pose.pose.orientation.z = q[3]
        msg.pose.pose.orientation.w = q[0]

        # Optional: Set small covariance to indicate high confidence
        msg.pose.covariance[0] = 0.1  # x
        msg.pose.covariance[7] = 0.1  # y
        msg.pose.covariance[35] = math.radians(5)**2  # yaw (in rad^2)

        self.initial_pose_pub.publish(msg)
        self.get_logger().info("📍 Published initial pose to AMCL")

    def amcl_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])[2]
        self.robot_pose = (x, y, yaw)

    def check_amcl_pose_timeout(self):
        if self.pose_sent and not self.robot_pose:
            if self.retry_attempts < self.max_retries:
                self.retry_attempts += 1
                self.get_logger().warn(f"🕒 AMCL pose not received yet. Retrying initial pose... (attempt {self.retry_attempts})")
                # Re-publish odometry-derived pose
                if self.odom_pose:
                    x, y, yaw = self.odom_pose
                    self.publish_initial_pose(x, y, math.degrees(yaw))
            else:
                self.get_logger().error("❌ Max retries reached. AMCL is not responding to initial pose.")

    def face_callback(self, msg: DetectedFace):
        new_pos = np.array([msg.position.x, msg.position.y])
        normal = (msg.normal.x, msg.normal.y)

        # Push position if too close to wall
        safe_pos = self.push_face_from_wall(new_pos)


        # Check if face is within 0.3m of any previously seen face
        for seen_pos in self.seen_faces:
            if np.linalg.norm(new_pos - np.array(seen_pos)) < 0.5:
                return  # Too close to a previously seen face

        # If it's a new one, add to seen
        self.seen_faces.add((msg.position.x, msg.position.y))

        # Convert face into pose
        face_pos = (msg.position.x, msg.position.y)

        # Prevent excessive closeness duplicates
        for pos, _ in self.face_queue:
            if math.hypot(pos[0] - safe_pos[0], pos[1] - safe_pos[1]) < 0.5:
                return

        if not np.all(np.isfinite(new_pos)) or not np.all(np.isfinite(normal)):
            self.get_logger().warn("Discarded invalid face with NaNs.")
            return

        self.face_queue.append((safe_pos, normal))
        self.publish_pushed_face_marker(safe_pos, normal)
        self.get_logger().info(f"👤 Received new face at ({new_pos})")

    def ring_callback(self, msg: DetectedRing):
        new_pos = np.array([msg.position.point.x, msg.position.point.y])
        color = msg.color.lower()

        for seen_pos in self.seen_rings:
            if np.linalg.norm(new_pos - np.array(seen_pos)) < 0.5:
                return

        self.seen_rings.add((msg.position.point.x, msg.position.point.y))
        ring_pos = (msg.position.point.x, msg.position.point.y)

        for pos, _ in self.ring_queue:
            if math.hypot(pos[0] - ring_pos[0], pos[1] - ring_pos[1]) < 0.5:
                return

        # Store as tuple with color
        self.ring_queue.append((ring_pos, color))
        # Store or log color as needed

        self.publish_ring_marker(new_pos)
        self.get_logger().info(f"🔔 Ring detected at {new_pos} with color '{color}'")

    def map_callback(self, msg):
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin.position

        # Convert occupancy grid to numpy array
        grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.occupancy = np.ones_like(grid)

        # Classify grid values
        self.occupancy[grid == 100] = 0    # Wall/obstacle
        self.occupancy[grid == 0] = 1      # Free space
        self.occupancy[grid == -1] = -1    # Unknown

    def markers_callback(self, msg):
        self.get_logger().info("📦 Received markers")
        self.visiting_point_camera_poses = []
        cam_map = {}

        for m in msg.markers:
            if m.ns == "inspection" and m.type == Marker.ARROW and m.color.b > 0.9:
                yaw = self.quaternion_to_yaw(m.pose.orientation)
                cam_map[m.id] = {
                    'pose': (m.pose.position.x, m.pose.position.y, yaw),
                    'targets': [],
                    'seen': set(),
                    'marker_id': m.id,
                    'hardcoded': m.id >= 10_000 
                }

        green_count = 0
        assigned_count = 0

        for m in msg.markers:
            if m.type == Marker.ARROW and m.color.g > 0.9 and m.ns == "inspection":
                # Extract normal direction from quaternion
                q = m.pose.orientation
                _, _, yaw = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])
                nx = math.cos(yaw)
                ny = math.sin(yaw)
                tx = m.pose.position.x
                ty = m.pose.position.y
                marker_id = m.id
                green_count += 1

                if cam_map:
                    closest_cam = min(
                        cam_map.values(),
                        key=lambda c: math.hypot(c['pose'][0] - tx, c['pose'][1] - ty)
                    )
                    closest_cam['targets'].append((tx, ty, nx, ny, marker_id))
                    assigned_count += 1

        self.get_logger().info(f"🟢 Total green markers: {green_count}, assigned to cameras: {assigned_count}")

        self.visiting_point_camera_poses = list(cam_map.values())
        self.get_logger().info(f"🟦 Loaded {len(self.visiting_point_camera_poses)} camera poses")

    def quaternion_to_yaw(self, q):
        return transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])[2]

    def speak(self, engine: Engine, text):
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.6)  # Volume level (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()

    def robot_state_loop(self):
        self.get_logger().info(f"Current Robot State - {self.robot_state}")
        #current_time = self.get_clock().now()
        #this function should only include handler calls and state transitions for clarity

        if self.robot_state == RobotState.INITIALIZING:
            robot_finished_initializing = self.handle_robot_initializing()

            if not robot_finished_initializing:
                self.robot_state = RobotState.INITIALIZING
            else:
                self.robot_state = RobotState.SELECTING_NEW_GOAL

        elif self.robot_state == RobotState.SELECTING_NEW_GOAL:

            selected_detected_face_to_visit = self.handle_robot_detected_face_selection()

            if selected_detected_face_to_visit:
                self.robot_state = RobotState.SERVICE_FACE_DETECTION
            elif self.handle_robot_detected_ring_selection():
                #function call in elif condition intentional - if both detected face and ring are available, popping both at the same time could pose issues 
                self.robot_state = RobotState.SERIVICE_RING_DETECTION
            else:
                new_goal_selected = self.handle_robot_selecting_new_inspection_goal()

                if new_goal_selected:
                    self.robot_state = RobotState.INSPECTING_GOAL
                else:
                    self.robot_state = RobotState.ROBOT_FINISHED

        elif self.robot_state == RobotState.INSPECTING_GOAL:

            goal_visited = self.handle_robot_inspecting_goal()

            if goal_visited:
                self.robot_state = RobotState.SELECTING_NEW_GOAL
            else:
                self.robot_state = RobotState.INSPECTING_GOAL

        elif self.robot_state == RobotState.SERVICE_FACE_DETECTION:

            finished_visiting_detected_face = self.handle_robot_visiting_face()
            
            if finished_visiting_detected_face:
                self.robot_state = RobotState.SELECTING_NEW_GOAL
            else:
                self.robot_state = RobotState.SERVICE_FACE_DETECTION

        elif self.robot_state == RobotState.SERIVICE_RING_DETECTION:

            finished_visiting_detected_ring = self.handle_robot_visiting_ring()

            if finished_visiting_detected_ring:
                self.robot_state = RobotState.SELECTING_NEW_GOAL
            else:
                self.robot_state = RobotState.SERIVICE_RING_DETECTION

        elif self.robot_state == RobotState.ROBOT_FINISHED:
            pass
        else:
            self.get_logger().warn(f"Illegal Robot State")
        

    def handle_robot_initializing(self):
        if self.robot_pose is None or self.occupancy is None or not self.visiting_point_camera_poses:
            return False
        return True
    
    def handle_robot_selecting_new_inspection_goal(self):
        if len(self.visiting_point_camera_poses) > 0:
            #select next goal
            next_goal = min(self.visiting_point_camera_poses, key=lambda c: self.astar_path_length(self.robot_pose, c['pose']))
            self.visiting_point_camera_poses.remove(next_goal)
            self.current_visiting_map_point = next_goal
            #add an empty set of seen green points
            if 'seen' not in self.current_visiting_map_point:
                self.current_visiting_map_point['seen'] = set()
            #move to new pose
            self.get_logger().info(f"➡️ Going to next pose at {self.current_visiting_map_point['pose']}")
            self.cmdr.goToPose(self.current_visiting_map_point['pose'])
            return True
        return False

    def handle_robot_inspecting_goal(self):
        finished_moving_to_pose = self.cmdr.isTaskComplete()

        #== SERVICE HARDCODED POINTS ==
        if self.current_visiting_map_point.get('hardcoded', False):
            if finished_moving_to_pose:
                self.get_logger().info("🐢🐢🐢 Arrived at hardcoded goal.")
                self.publish_visited_markers(self.current_visiting_map_point)
                return True
            return False

        #== SERVICE NORMAL POINTS ==
        
        #mark green markers as seen
        for i, (tx, ty, nx, ny, _) in enumerate(self.current_visiting_map_point['targets']):
            if i in self.current_visiting_map_point['seen']:
                continue
            if self.is_visible(self.robot_pose, (tx, ty), (nx, ny)):
                self.current_visiting_map_point['seen'].add(i)

        #all green markers seen - cancel move and mark visiting point as visited
        #if len(self.current_visiting_map_point['seen']) == len(self.current_visiting_map_point['targets']):
        #    self.cmdr.cancelTask()
        #    self.publish_visited_markers(self.current_visiting_map_point)
        #    self.get_logger().info("✅ All targets seen. Canceling move.")
        #    return True
        
        #not all green markers seen
        if finished_moving_to_pose:
            self.get_logger().info("🏁 Arrived at goal.")
            self.publish_visited_markers(self.current_visiting_map_point)
            return True
        
        #still moving to pose
        return False
        
    def handle_robot_detected_face_selection(self):
        if self.face_queue:
            face = self.face_queue.popleft()
            x, y = face[0]
            yaw = math.atan2(-face[1][1], -face[1][0])

            self.cmdr.goToPose((x, y, yaw))
            self.get_logger().info(f"🧠 Navigating to detected face at {face}")
            return True
        return False

    def handle_robot_detected_ring_selection(self):
        if self.ring_queue:
            ring, color = self.ring_queue.popleft()
            #self.interrupt_start_time = now

            rx, ry, _ = self.robot_pose
            tx, ty = ring
            yaw = math.atan2(ty - ry, tx - rx)
            self.cmdr.goToPose((tx, ty, yaw))

            self.ring_color = color
            self.get_logger().info(f"🟡 Navigating to ring at {ring} (color: {color})")
            return True
        return False

    def handle_robot_visiting_face(self):
        finished_visiting_face = self.cmdr.isTaskComplete()

        if finished_visiting_face:
            self.speak(self.tts_engine, f"Hello Persons")
            self.get_logger().info("✅ Finished visiting face")
            return True
        return False

    def handle_robot_visiting_ring(self):
        rx, ry, _ = self.robot_pose
        tx, ty = self.active_ring_goal
        dist = math.hypot(tx - rx, ty - ry)

        finished_visiting_ring = self.cmdr.isTaskComplete()
        in_ring_proximity = dist < self.ring_visit_dist

        #timeout_elapsed = (
        #    hasattr(self, "interrupt_start_time") and 
        #    (now - self.interrupt_start_time).nanoseconds > 20 * 1e9  # 20 seconds
        #)

        if finished_visiting_ring or in_ring_proximity:
            self.cmdr.cancelTask()
            self.speak(self.tts_engine, f"This is a {self.ring_color} ring")
            self.ring_color = None

            self.get_logger().info("✅ Reached or finished ring. Canceling goal and resuming inspection.")
            return True
        return False


    def is_visible(self, robot_pose, target, normal, fov_deg=90, min_angle_deg=45):
        if self.occupancy is None or self.resolution is None or self.origin is None:
            return False

        rx, ry, ryaw = robot_pose
        tx, ty = target
        nx, ny = normal

        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        #skip = False
        if dist == 0:
            #skip = True
            return False

        # 🔍 FIELD OF VIEW CHECK
        view_angle = math.atan2(dy, dx)
        angle_to_heading = abs((ryaw - view_angle + math.pi) % (2 * math.pi) - math.pi)

        if angle_to_heading > math.radians(fov_deg / 2):
            #skip = True
            return False

        # 📏 NORMAL ANGLE CHECK
        heading_x = math.cos(ryaw)
        heading_y = math.sin(ryaw)

        norm_len = math.hypot(nx, ny)
        if norm_len == 0:
            #skip = True
            angle = 999.0
            return False
        else:
            nx /= norm_len
            ny /= norm_len

            # Flip normal (we want to face *into* it)
            dot = heading_x * -nx + heading_y * -ny
            cos_angle = max(min(dot, 1.0), -1.0)
            angle = math.acos(cos_angle)

        if angle > math.radians(min_angle_deg):
            #skip = True
            return False

        # 🧱 LINE OF SIGHT CHECK (Bresenham)
        rx_pix = int((rx - self.origin.x) / self.resolution)
        ry_pix = int((ry - self.origin.y) / self.resolution)
        tx_pix = int((tx - self.origin.x) / self.resolution)
        ty_pix = int((ty - self.origin.y) / self.resolution)

        #los = True
        for x, y in self.bresenham(rx_pix, ry_pix, tx_pix, ty_pix):
            if 0 <= x < self.occupancy.shape[1] and 0 <= y < self.occupancy.shape[0]:
                if self.occupancy[y, x] != 1:
                    #los = False
                    #skip = True
                    return False

        #self.get_logger().info(f"Target pos = ({tx:.1f}, {ty:.1f}), FOV angle = {math.degrees(angle_to_heading):.1f}°, Facing angle = {math.degrees(angle):.1f}°, LOS = {los}")

        return True #not skip

    def bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        err = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def publish_visited_markers(self, cam):
        # Publishes the finished position and all targets that were seen for that position
        ma = MarkerArray()
        for i in cam['seen']:
            tx, ty, *_ = cam['targets'][i]
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "visited"
            m.id = int(tx * 100) + int(ty * 100)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = tx
            m.pose.position.y = ty
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        x, y, _ = cam['pose']
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "visited"
        m.id = int(x * 1000) + int(y * 1000) + 999999
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.1
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        ma.markers.append(m)

        self.pub_visited.publish(ma)

    def publish_ring_marker(self, position):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rings"
        marker.id = int(position[0] * 1000) + int(position[1] * 1000)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.pub_ring_marker.publish(MarkerArray(markers=[marker]))

    def publish_pushed_face_marker(self, position, normal=None):
        # Red dot for the face position
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "pushed_faces"
        m.id = int(position[0] * 1000) + int(position[1] * 1000)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = position[0]
        m.pose.position.y = position[1]
        m.pose.position.z = 0.0
        m.scale.x = 0.15
        m.scale.y = 0.15
        m.scale.z = 0.05
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        self.pushed_face_pub.publish(m)

        # Optional arrow for the normal vector
        if normal is not None:
            arrow = Marker()
            arrow.header.frame_id = "map"
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns = "pushed_faces"
            arrow.id = m.id + 1000000
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.05  # shaft diameter
            arrow.scale.y = 0.1   # head diameter
            arrow.scale.z = 0.1   # head length
            arrow.color.r = 0.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0

            start = position
            end = (
                position[0] + normal[0] * 0.5,
                position[1] + normal[1] * 0.5,
                0.0
            )
            arrow.points.append(self.make_point(start))
            arrow.points.append(self.make_point(end))

            self.pushed_face_pub.publish(arrow)

    def make_point(self, pos):
        pt = PointStamped().point
        pt.x = pos[0]
        pt.y = pos[1]
        pt.z = pos[2] if len(pos) > 2 else 0.0
        return pt

    def astar_path_length(self, p1, p2):
        if self.occupancy is None or self.resolution is None:
            return float('inf')

        start = (int((p1[0] - self.origin.x) / self.resolution), int((p1[1] - self.origin.y) / self.resolution))
        goal = (int((p2[0] - self.origin.x) / self.resolution), int((p2[1] - self.origin.y) / self.resolution))

        return self.astar(start, goal, self.occupancy)

    def astar(self, start, goal, grid):
        height, width = grid.shape
        visited = set()
        queue = [(0 + self.heuristic(start, goal), 0, start)]
        g_score = {start: 0}

        while queue:
            _, cost, current = heapq.heappop(queue)
            if current == goal:
                return cost
            if current in visited:
                continue
            visited.add(current)

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                if grid[neighbor[1], neighbor[0]] != 1:
                    continue

                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(queue, (priority, tentative_g, neighbor))

        return float('inf')

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def distance_to_nearest_wall(self, pos, search_radius=0.5):
        """
        Estimates distance from position to the nearest obstacle in the occupancy grid.
        """
        gx = int((pos[0] - self.origin.x) / self.resolution)
        gy = int((pos[1] - self.origin.y) / self.resolution)
        radius_px = int(search_radius / self.resolution)

        min_dist = float('inf')
        for dx in range(-radius_px, radius_px + 1):
            for dy in range(-radius_px, radius_px + 1):
                x = gx + dx
                y = gy + dy
                if 0 <= x < self.occupancy.shape[1] and 0 <= y < self.occupancy.shape[0]:
                    if self.occupancy[y, x] != 1:
                        dist = math.hypot(dx, dy) * self.resolution
                        if dist < min_dist:
                            min_dist = dist

        return min_dist

    def push_face_from_wall(self, pos, min_dist=0.3, max_push=0.5, step=0.05):
        """
        Push the face away from the closest obstacle by checking around it and computing the direction
        from the nearest wall cell to the face.
        """
        if self.occupancy is None or self.resolution is None or self.origin is None:
            return pos  # Fallback if map is not available

        gx = int((pos[0] - self.origin.x) / self.resolution)
        gy = int((pos[1] - self.origin.y) / self.resolution)
        radius_px = int(max_push / self.resolution)

        height, width = self.occupancy.shape
        nearest_obs = None
        min_d2 = float('inf')

        # 🔍 Find nearest obstacle in the surrounding area
        for dx in range(-radius_px, radius_px + 1):
            for dy in range(-radius_px, radius_px + 1):
                x = gx + dx
                y = gy + dy
                if 0 <= x < width and 0 <= y < height:
                    if self.occupancy[y, x] != 1:
                        d2 = dx**2 + dy**2
                        if d2 < min_d2:
                            min_d2 = d2
                            nearest_obs = (x, y)

        if nearest_obs is None:
            return pos  # No obstacle nearby

        # Compute push direction
        obs_world = (
            self.origin.x + nearest_obs[0] * self.resolution,
            self.origin.y + nearest_obs[1] * self.resolution,
        )

        push_dir = np.array(pos) - np.array(obs_world)
        if np.linalg.norm(push_dir) == 0:
            return pos  # Cannot compute direction

        push_dir /= np.linalg.norm(push_dir)
        current_pos = np.array(pos)

        # Push outward until we're >= min_dist from wall
        while self.distance_to_nearest_wall(current_pos) < min_dist and np.linalg.norm(current_pos - pos) < max_push:
            current_pos += push_dir * step

            # Ensure still in free space
            gx = int((current_pos[0] - self.origin.x) / self.resolution)
            gy = int((current_pos[1] - self.origin.y) / self.resolution)
            if not (0 <= gx < width and 0 <= gy < height) or self.occupancy[gy, gx] != 1:
                return pos  # Invalid or blocked

        return tuple(current_pos)

def main(args=None):
    rclpy.init(args=args)
    node = InspectionNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

