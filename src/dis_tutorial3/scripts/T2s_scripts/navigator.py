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
from geometry_msgs.msg import Point
import time
import subprocess
import json
import importlib.resources
import os
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from robot_commander import RobotCommander
from dis_tutorial3.msg import DetectedFace
from dis_tutorial3.msg import DetectedRing
from dis_tutorial3.msg import DetectedBird
from T2s_custom_modules.dialogue import *
import T2s_custom_modules.dialogue_start_process

from dis_tutorial3.msg import BridgePose

from geometry_msgs.msg import PoseWithCovarianceStamped

import pyttsx3
from pyttsx3.engine import Engine

from enum import Enum, auto

class RobotState(Enum):
    INITIALIZING = auto()
    SELECTING_NEW_GOAL = auto()
    INSPECTING_GOAL = auto()
    SELECTING_PERSON = auto()
    SERVICE_CONVERSATION = auto()
    MOVING_TO_BRIDGE = auto()
    BRIDGE_NAVIGATION = auto()
    GO_TO_FINAL_POSITION = auto()
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
        self.pub_bird_marker = self.create_publisher(MarkerArray, '/bird_markers', 10)

        self.robot_state_pub = self.create_publisher(String, '/robot_internal_state', 10)

        self.arm_command_pub = self.create_publisher(String, "/arm_command", 10)

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

        self.create_subscription(
            DetectedBird,
            '/detected_birds',
            self.bird_callback,
            qos_profile_sensor_data
        )


        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        self.bridge_pose_sub = self.create_subscription(
            BridgePose,
            "/bridge_pose_map",
            self.bridge_pose_callback,
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
        self.bird_queue = deque()
        self.seen_faces = set()
        self.seen_rings = set()
        self.seen_birds = set()

        self.bird_data = {}  # {(x, y): {'classifications': [str], 'visited': bool}}

        self.current_face = None

        #self.ring_color = None
        #self.visit_ring_position =None
        #self.ring_visit_dist = 0.6
        
        
        #bridge navigation
        self.bridge_start_position = (0.00, -0.80, -np.pi/2)
        self.red_parking_position = (1.24, -6.74, 0.0)
        self.latest_brige_position = None
        self.moving_to_brige_pose = None

        #start up everything
        self.tts_engine = pyttsx3.init()
        self.cmdr = RobotCommander()

        loop_update_delay_seconds = 0.3
        self.timer = self.create_timer(loop_update_delay_seconds, self.robot_state_loop)

    def publish_robot_state(self):
        msg = String()
        msg.data = str(self.robot_state.name)
        self.robot_state_pub.publish(msg)

    def arm_position_ring_bird_search(self):
        msg = String()
        yaw = 0.0
        link1_rotation = 0.0
        link2_rotation = 0.0
        link3_rotation = np.pi / 2
        msg.data = f"manual:[{yaw},{link1_rotation},{link2_rotation},{link3_rotation}]"
        self.arm_command_pub.publish(msg)
        self.get_logger().info("Published Arm position for bird search")

    def arm_position_bridge_nav(self):
        msg = String()
        angle_offset = math.radians(15)
        base_link_bend = 0.3
        bend_factor = 0.3
        yaw = 0.0
        link1_rotation = base_link_bend
        link2_rotation = bend_factor - base_link_bend
        link3_rotation = np.pi - link1_rotation - link2_rotation - angle_offset
        msg.data = f"manual:[{yaw},{link1_rotation},{link2_rotation},{link3_rotation}]"
        self.arm_command_pub.publish(msg)
        self.get_logger().info("Published Arm position for bridge navigation")

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
        gender = msg.gender

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
        for pos, _, _ in self.face_queue:
            if math.hypot(pos[0] - safe_pos[0], pos[1] - safe_pos[1]) < 0.5:
                return

        if not np.all(np.isfinite(new_pos)) or not np.all(np.isfinite(normal)):
            self.get_logger().warn("Discarded invalid face with NaNs.")
            return

        self.face_queue.append((safe_pos, normal, gender))
        self.publish_pushed_face_marker(safe_pos, normal, gender)
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

        self.publish_ring_marker(new_pos, color)
        self.get_logger().info(f"🔔 Ring detected at {new_pos} with color '{color}'")

    def bird_callback(self, msg: DetectedBird):
        pos = (msg.position.x, msg.position.y)
        class_name = msg.class_name.lower()

        if not np.all(np.isfinite(pos)):
            self.get_logger().warn("Discarded invalid bird with NaNs.")
            return

        existing_pos = None
        for known_pos in self.bird_data:
            if np.linalg.norm(np.array(pos) - np.array(known_pos)) < 0.5:
                existing_pos = known_pos
                break

        if existing_pos:
            # Update classification and confidence
            if not self.bird_data[existing_pos]['visited']:
                self.bird_data[existing_pos]['class_name'] = class_name
                self.bird_data[existing_pos]['confidence'] = msg.confidence
        else:
            # Add new entry
            self.bird_data[pos] = {
                'class_name': class_name,
                'confidence': msg.confidence,
                'visited': False
            }
            self.bird_queue.append((pos, class_name))
            self.publish_bird_marker(pos, class_name)
            self.speak(f"{class_name}")
            self.get_logger().info(f"🕊️ New bird found at {pos} ({class_name})")

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

    def speak(self, text):
        self.tts_engine.setProperty('rate', 120)
        self.tts_engine.setProperty('volume', 0.6)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        time.sleep(0.2) 

    def bridge_pose_callback(self, msg: BridgePose):
        x = msg.pose.position.x
        y = msg.pose.position.y
        relative_yaw = self.quaternion_to_yaw(msg.pose.orientation)
        is_final_position = msg.is_final_position  # <--- The boolean flag

        if self.robot_pose is None:
            self.get_logger().warn("⚠️ Received bridge pose but AMCL pose is not yet available.")
            return

        _, _, robot_yaw = self.robot_pose
        global_yaw = self.normalize_angle_rad(robot_yaw - relative_yaw)

        self.latest_brige_position = (x, y, global_yaw, is_final_position)

    def robot_state_loop(self):
        self.get_logger().info(f"Current Robot State - {self.robot_state}")
        #current_time = self.get_clock().now()
        #this function should only include handler calls and state transitions for clarity

        if self.robot_state == RobotState.INITIALIZING:
            robot_finished_initializing = self.handle_robot_initializing()

            if not robot_finished_initializing:
                self.robot_state = RobotState.INITIALIZING
            else:
                self.arm_position_ring_bird_search()
                self.robot_state = RobotState.SELECTING_NEW_GOAL 
                #self.arm_position_bridge_nav()
                #self.robot_state = RobotState.MOVING_TO_BRIDGE               

        elif self.robot_state == RobotState.SELECTING_NEW_GOAL:

            new_goal_selected = self.handle_robot_selecting_new_inspection_goal()

            if new_goal_selected:
                self.robot_state = RobotState.INSPECTING_GOAL

                #if self.face_queue:
                #    self.robot_state = RobotState.SELECTING_PERSON

            else:
                self.robot_state = RobotState.SELECTING_PERSON

        elif self.robot_state == RobotState.INSPECTING_GOAL:

            goal_visited = self.handle_robot_inspecting_goal()

            if goal_visited:
                self.robot_state = RobotState.SELECTING_NEW_GOAL
            else:
                self.robot_state = RobotState.INSPECTING_GOAL

        elif self.robot_state == RobotState.SELECTING_PERSON:

            selected_a_face = self.handle_robot_detected_face_selection()

            if selected_a_face:
                self.robot_state = RobotState.SERVICE_CONVERSATION
            else:
                self.arm_position_bridge_nav()
                self.robot_state = RobotState.MOVING_TO_BRIDGE

        elif self.robot_state == RobotState.SERVICE_CONVERSATION:

            finished_visiting_detected_face = self.handle_robot_visiting_face()

            if finished_visiting_detected_face:
                self.robot_state = RobotState.SELECTING_PERSON
            else:
                self.robot_state = RobotState.SERVICE_CONVERSATION

        elif self.robot_state == RobotState.MOVING_TO_BRIDGE:
            done_moving_to_bridge_start = self.handle_robot_moving_to_bridge()

            if done_moving_to_bridge_start:

                self.robot_state = RobotState.BRIDGE_NAVIGATION
            else:
                self.robot_state = RobotState.MOVING_TO_BRIDGE

        elif self.robot_state == RobotState.BRIDGE_NAVIGATION:

            crossed_bridge = self.handle_robot_bridge_navigation()

            if crossed_bridge:
                self.robot_state = RobotState.GO_TO_FINAL_POSITION
            else:
                self.robot_state = RobotState.BRIDGE_NAVIGATION

        elif self.robot_state == RobotState.GO_TO_FINAL_POSITION:
            done_parking = self.handle_robot_moving_to_parking()

            if done_parking:
                self.robot_state = RobotState.ROBOT_FINISHED
            else:
                self.robot_state = RobotState.GO_TO_FINAL_POSITION

        elif self.robot_state == RobotState.ROBOT_FINISHED:
            pass

        else:
            self.get_logger().warn(f"Illegal Robot State")

        self.publish_robot_state()


    def handle_robot_moving_to_parking(self):
        rx, ry, ryaw = self.robot_pose

        return True

        #self.move_to_position(self.red_parking_position)
        return False
    
    def move_to_position(self, bridge_pose):
        self.cmdr.goToPose(bridge_pose)

    def normalize_angle_rad(self, angle_rad):
        return (angle_rad + math.pi) % (2 * math.pi) - math.pi

    def handle_robot_moving_to_bridge(self):
        rx, ry, ryaw = self.robot_pose               # ryaw in radians
        tx, ty, tyaw = self.bridge_start_position     # tyaw in radians

        dist_to_start = math.hypot(tx - rx, ty - ry)
        yaw_error = self.normalize_angle_rad(tyaw - ryaw)

        dist_thresh_in_meters = 0.08
        angle_thresh_in_radians = math.radians(5)  # ~0.087 radians

        if dist_to_start < dist_thresh_in_meters and abs(yaw_error) < angle_thresh_in_radians:
            return True

        self.move_to_position(self.bridge_start_position)
        return False
        
    def handle_robot_bridge_navigation(self):
        rx, ry, ryaw = self.robot_pose

        done_with_bridge_move = self.cmdr.isTaskComplete()

        # Aligning yaw if we reached the previous goal
        if done_with_bridge_move and self.latest_brige_position is not None and self.moving_to_brige_pose is not None:
            mtx, mty, goal_yaw, is_final = self.moving_to_brige_pose
            yaw_error = self.normalize_angle_rad(goal_yaw - ryaw)

            #RED = '\033[91m'
            #RESET = '\033[0m'
            #self.get_logger().info(f"{RED}[Bridge Nav] Computed goal yaw: {math.degrees(goal_yaw):.1f}°{RESET}")
            #self.get_logger().info(f"{RED}[Current Pose] Robot yaw: {math.degrees(ryaw):.1f}°{RESET}")
            #self.get_logger().info(f"{RED}[Yaw Error] goal - current = {math.degrees(yaw_error):.1f}°{RESET}")

            if abs(yaw_error) > math.radians(7):
                #CYAN = '\033[96m'
                #RESET = '\033[0m'
                #self.get_logger().warning(
                #    f"{CYAN}Yaw misaligned (current: {math.degrees(ryaw):.1f}°, target: {math.degrees(goal_yaw):.1f}°, error: {math.degrees(yaw_error):.1f}°) → retrying same pose{RESET}"
                #)
                pos = (mtx, mty, goal_yaw)
                self.move_to_position(pos)
                return False
            #done
            if is_final:
                return True


            # Aligned: go into "waiting for new bridge pose" mode
            self.get_logger().info("Yaw aligned. Waiting for new bridge position...")
            self.moving_to_brige_pose = None
            self.last_used_bridge_pose = self.latest_brige_position  # Mark it as used



        # New bridge target received and ready to move
        if done_with_bridge_move and self.latest_brige_position is not None and self.moving_to_brige_pose is None:
            if self.latest_brige_position != getattr(self, "last_used_bridge_pose", None):
                #self.speak("New Bridge Point")
                #self.get_logger().warning(
                #    f"Robot with YAW - {math.degrees(ryaw):.1f}° - Going To New Bridge Pos {self.latest_brige_position}"
                #)
                lbx, lby, lbyaw, lbfinal = self.latest_brige_position
                pos = (lbx, lby, lbyaw)

                self.move_to_position(pos)
                self.moving_to_brige_pose = self.latest_brige_position
            else:
                #self.get_logger().info("Waiting for new bridge position...")
                pass
        elif self.latest_brige_position is None:
            #self.get_logger().warn("No bridge position available")
            pass

        return False

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
            self.current_face = face
            return True
        return False

    def handle_robot_visiting_face(self):
        got_to_face = self.cmdr.isTaskComplete()
        if not got_to_face:
            return False

        if self.current_face is None:
            # failsafe
            return True

        gender = "MAN" if self.current_face[2] == "man" else "WOMAN"
        birds = list(self.bird_queue)
        rings = list(self.ring_queue)


        dialogue_script_path = T2s_custom_modules.dialogue_start_process.__file__
        pkg_dir = os.path.dirname(dialogue_script_path)
        lib_dir = os.path.dirname(pkg_dir)
        self.get_logger().info(f"Launching dialogue script at: {dialogue_script_path}")

        env = os.environ.copy()
        env["PYTHONPATH"] = lib_dir + ":" + env.get("PYTHONPATH", "")

        # Run the dialogue GUI as a blocking subprocess
        try:
            subprocess.run(
                [
                    "python3", "-m", "T2s_custom_modules.dialogue_start_process",
                    gender,
                    json.dumps(rings),
                    json.dumps(birds),
                ],
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Dialogue script failed: {e}")

        self.current_face = None
        return True
    
    # def handle_robot_detected_ring_selection(self):
    #     if self.ring_queue:
    #         ring, color = self.ring_queue.popleft()
            
    #         rx, ry, _ = self.robot_pose
    #         tx, ty = ring
    #         yaw = math.atan2(ty - ry, tx - rx)
    #         self.cmdr.goToPose((tx, ty, yaw))

    #         self.ring_color = color
    #         self.visit_ring_position = ring
    #         self.get_logger().info(f"🟡 Navigating to ring at {ring} (color: {color})")
    #         return True
    #     return False

    # def handle_robot_visiting_ring(self):
    #     rx, ry, _ = self.robot_pose
    #     tx, ty = self.visit_ring_position
    #     dist = math.hypot(tx - rx, ty - ry)

    #     finished_visiting_ring = self.cmdr.isTaskComplete()
    #     in_ring_proximity = dist < self.ring_visit_dist
    #     self.get_logger().info(f"IN RING PROXIMITY - {in_ring_proximity}")

    #     #timeout_elapsed = (
    #     #    hasattr(self, "interrupt_start_time") and 
    #     #    (now - self.interrupt_start_time).nanoseconds > 20 * 1e9  # 20 seconds
    #     #)

    #     if finished_visiting_ring or in_ring_proximity:
    #         self.cmdr.cancelTask()
    #         self.speak(f"This is a {self.ring_color} ring")
    #         self.ring_color = None
    #         self.visit_ring_position = None

    #         self.get_logger().info("✅ Reached or finished ring. Canceling goal and resuming inspection.")
    #         return True
    #     return False


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


    def publish_ring_marker(self, position, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rings"
        marker.id = int(position[0] * 1000) + int(position[1] * 1000)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.scale.x = 0.05  # ring thickness

        # Color mapping
        color_map = {
            "black": (0.05, 0.05, 0.05, 1.0),
            "red":   (1.0, 0.1, 0.1, 1.0),
            "blue":  (0.2, 0.5, 1.0, 1.0),
            "green": (0.2, 1.0, 0.3, 1.0)
        }
        color_rgba = color_map.get(color.lower(), (1.0, 1.0, 0.0, 1.0))  # yellow fallback
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgba

        # Circle points for the ring
        ring_radius = 0.09
        num_points = 30
        for i in range(num_points + 1):  # +1 to close the loop
            angle = 2 * math.pi * i / num_points
            pt = Point()
            pt.x = position[0] + ring_radius * math.cos(angle)
            pt.y = position[1] + ring_radius * math.sin(angle)
            pt.z = position[2] if len(position) > 2 else 0.0
            marker.points.append(pt)

        self.pub_ring_marker.publish(MarkerArray(markers=[marker]))


    def publish_bird_marker(self, position, class_name=""):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "birds"
        marker.id = int(position[0] * 1000) + int(position[1] * 1000)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.05
        # Light blue/cyan
        marker.color.r = 0.3
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Label marker (red text)
        label = Marker()
        label.header.frame_id = "map"
        label.header.stamp = marker.header.stamp
        label.ns = "birds"
        label.id = marker.id + 1000000
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = position[0] + 0.15
        label.pose.position.y = position[1]
        label.pose.position.z = 0.2
        label.scale.z = 0.12  # text height
        # Red text
        label.color.r = 1.0
        label.color.g = 0.1
        label.color.b = 0.1
        label.color.a = 1.0
        label.text = class_name

        self.pub_bird_marker.publish(MarkerArray(markers=[marker, label]))



    def publish_pushed_face_marker(self, position, normal=None, gender=None):
        if gender == "man":
            color_r, color_g, color_b, color_a = 0.1, 0.1, 0.8, 1.0  # dark blue
        elif gender == "woman":
            color_r, color_g, color_b, color_a = 1.0, 0.2, 0.8, 1.0  # pink
        else:
            color_r, color_g, color_b, color_a = 0.8, 0.8, 0.8, 1.0  # gray for unknown

        # Red dot for the face position (now gender-based)
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
        m.color.r = color_r
        m.color.g = color_g
        m.color.b = color_b
        m.color.a = color_a
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
            arrow.color.r = color_r
            arrow.color.g = color_g
            arrow.color.b = color_b
            arrow.color.a = color_a

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

