#!/usr/bin/env python3

import carla
import math
import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Range

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

class Ros2BridgeAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file=None):
        super().__init__(path_to_conf_file)
        print("Ros2BridgeAgent initialized")
        
        # Agent state
        self.vehicle = None
        self.sensor_list = []
        self.front_distance = float('inf')
        self.waypoints = []
        self.current_waypoint_index = 0
        self.lookahead_distance = 8.0
        self.previous_steering = 0.0
        self.lane_change_state = "none"
        self.lane_change_start_time = None
        self.left_turn_duration = 2.0
        self.straight_duration = 17.0  # Set to 20 seconds as requested
        self.right_turn_duration = 2.0
        self._last_step_time = None
        
        # Lane change enable control
        self._lane_change_enabled = False
        self._lane_change_enabled_time = 0
        self._startup_delay = 13.0
        
        # Lane change cooldown and completion control
        self._lane_change_cooldown = 0.0
        self._cooldown_duration = 5.0
        self._overtaking_completed = False
        
        # Adversary vehicle tracking (used for logging only, not triggering)
        self.adversary_vehicle = None
        self.adversary_speed_kmh = 0.0
        self.vehicle_detection_distance = 50.0
        self.same_lane_threshold = 2.0
        self.lane_change_distance_threshold = 10.0
        
        # Spectator view settings
        self.spectator = None
        self.spectator_height = 8.0
        self.spectator_distance = 10.0
        
        # Fixed speed
        self.target_speed = 45.0 / 3.6  # ~40 km/h as previously set
        
        # Initialize ROS2
        try:
            rclpy.init(args=None)
            self.node = rclpy.create_node('ros2_bridge_agent')
            print("ROS2 node created successfully")
            
            self.status_pub = self.node.create_publisher(String, '/carla/status', 10)
            self.radar_pub = self.node.create_publisher(Range, '/carla/radar', 10)
            
            self.ros_thread = threading.Thread(target=self.spin_ros, daemon=True)
            self.ros_thread.start()
            
        except Exception as e:
            print(f"ROS2 initialization error: {e}")
            self.node = None
    
    def sensors(self):
        print("Sensors method called")
        return []
    
    def spin_ros(self):
        try:
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.05)
                time.sleep(0.05)
        except Exception as e:
            print(f"ROS2 spin error: {e}")
    
    def execute_lane_change(self, direction, timestamp):
        print(f"=== INITIATING {direction.upper()} LANE CHANGE MANEUVER ===")
        
        vehicle_loc = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        target_lane = current_waypoint.get_left_lane() if direction == "left" else current_waypoint.get_right_lane()
        if not target_lane or target_lane.lane_type != carla.LaneType.Driving:
            print(f"WARNING: No valid {direction} lane available for lane change!")
            return False
        
        self.lane_change_state = direction
        self.lane_change_start_time = timestamp
        return True
    
    def generate_path(self, distance=500, lane_change=None):
        if not self.vehicle or not self.world:
            return []
        vehicle_loc = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        print(f"Initial waypoint: road_id={current_waypoint.road_id}, lane_id={current_waypoint.lane_id}, " +
              f"has_left_lane={current_waypoint.get_left_lane() is not None}, " +
              f"has_right_lane={current_waypoint.get_right_lane() is not None}")
        
        target_waypoint = current_waypoint
        if lane_change == 'left' and current_waypoint.get_left_lane():
            target_waypoint = current_waypoint.get_left_lane()
            if target_waypoint.lane_type != carla.LaneType.Driving:
                print("WARNING: Left lane is not a driving lane, staying in current lane")
                target_waypoint = current_waypoint
            else:
                print(f"Switched to LEFT lane: road_id={target_waypoint.road_id}, lane_id={target_waypoint.lane_id}")
        elif lane_change == 'right' and current_waypoint.get_right_lane():
            target_waypoint = current_waypoint.get_right_lane()
            if target_waypoint.lane_type != carla.LaneType.Driving:
                print("WARNING: Right lane is not a driving lane, staying in current lane")
                target_waypoint = current_waypoint
            else:
                print(f"Switched to RIGHT lane: road_id={target_waypoint.road_id}, lane_id={target_waypoint.lane_id}")
        
        self.waypoints = [target_waypoint]
        distance_traveled = 0
        waypoint_separation = 2.0
        current_waypoint = target_waypoint
        
        while distance_traveled < distance:
            next_waypoints = current_waypoint.next(waypoint_separation)
            if not next_waypoints:
                print("WARNING: No next waypoints found, path generation stopping early")
                break
            current_waypoint = next_waypoints[0]
            self.waypoints.append(current_waypoint)
            distance_traveled += waypoint_separation
        
        self.current_waypoint_index = 0
        lane_desc = 'left' if lane_change == 'left' else 'right' if lane_change == 'right' else 'current'
        print(f"Generated {len(self.waypoints)} waypoints covering {distance_traveled:.1f}m in {lane_desc} lane")
        return self.waypoints
    
    def update_waypoints(self):
        if not self.waypoints:
            return
        vehicle_loc = self.vehicle.get_location()
        min_distance = float('inf')
        closest_index = self.current_waypoint_index
        for i in range(self.current_waypoint_index, len(self.waypoints)):
            waypoint = self.waypoints[i]
            distance = vehicle_loc.distance(waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        self.current_waypoint_index = max(closest_index, self.current_waypoint_index)
        if self.current_waypoint_index >= len(self.waypoints) - 20:
            self.generate_path(distance=500)
    
    def get_target_waypoint(self):
        self.update_waypoints()
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return None
        vehicle_loc = self.vehicle.get_location()
        cumulative_distance = 0
        target_index = self.current_waypoint_index
        for i in range(self.current_waypoint_index + 1, len(self.waypoints)):
            waypoint = self.waypoints[i]
            prev_waypoint = self.waypoints[i-1]
            segment_distance = waypoint.transform.location.distance(prev_waypoint.transform.location)
            cumulative_distance += segment_distance
            if cumulative_distance >= self.lookahead_distance:
                target_index = i
                break
        if target_index >= len(self.waypoints):
            target_index = len(self.waypoints) - 1
        return self.waypoints[target_index]
    
    def calculate_steering(self):
        target_waypoint = self.get_target_waypoint()
        if not target_waypoint:
            return 0.0
        vehicle_transform = self.vehicle.get_transform()
        target_loc = target_waypoint.transform.location
        vehicle_loc = vehicle_transform.location
        direction_vector = target_loc - vehicle_loc
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_right = vehicle_transform.get_right_vector()
        forward_dot = direction_vector.x * vehicle_forward.x + direction_vector.y * vehicle_forward.y
        right_dot = direction_vector.x * vehicle_right.x + direction_vector.y * vehicle_right.y
        steering_angle = math.atan2(right_dot, forward_dot)
        
        if self.lane_change_state in ["left", "right"]:
            smoothed_steering = 0.5 * self.previous_steering + 0.5 * steering_angle
        else:
            smoothed_steering = 0.7 * self.previous_steering + 0.3 * steering_angle
            
        self.previous_steering = smoothed_steering
        
        if abs(smoothed_steering) > 0.2:
            print(f"Significant steering: {smoothed_steering:.3f}, target_pos=({target_loc.x:.1f}, {target_loc.y:.1f}), ego_pos=({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f})")
            
        return max(-1.0, min(1.0, smoothed_steering))
    
    def initialize(self):
        try:
            self.vehicle = CarlaDataProvider.get_hero_actor()
            if not self.vehicle:
                print("Failed to get ego vehicle")
                return
            print(f"Found ego vehicle with ID {self.vehicle.id}")
            self.vehicle.set_simulate_physics(True)
            self.world = CarlaDataProvider.get_world()
            
            self.spectator = self.world.get_spectator()
            if self.spectator:
                print("Spectator view set to follow ego vehicle")
            
            vehicle_loc = self.vehicle.get_location()
            current_waypoint = self.world.get_map().get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            print(f"Initial road: road_id={current_waypoint.road_id}, lane_id={current_waypoint.lane_id}, " +
                  f"has_left_lane={current_waypoint.get_left_lane() is not None}")
            
            self.generate_path(distance=500)
            self.create_sensors()
        except Exception as e:
            print(f"Initialization error: {e}")
    
    def create_sensors(self):
        if not self.vehicle:
            return
        try:
            world = self.world
            blueprint_library = world.get_blueprint_library()
            
            radar_bp = blueprint_library.find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', '30')
            radar_bp.set_attribute('vertical_fov', '30')
            radar_bp.set_attribute('range', '100')
            radar_bp.set_attribute('points_per_second', '2000')
            
            radar_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=1.0))
            radar = world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
            
            radar.listen(lambda data: self.radar_callback(data))
            self.sensor_list.append(radar)
            print(f"Created enhanced radar sensor")
        except Exception as e:
            print(f"Error creating sensors: {e}")
    
    def radar_callback(self, radar_data):
        try:
            closest_dist = float('inf')
            closest_speed = 0.0
            
            for detection in radar_data:
                azimuth = math.degrees(detection.azimuth)
                if abs(azimuth) < 15:
                    distance = detection.depth
                    if distance < closest_dist:
                        closest_dist = distance
                        closest_speed = detection.velocity
            
            self.front_distance = closest_dist if closest_dist < float('inf') else 100.0
            
            if closest_speed < 0:
                ego_velocity = self.vehicle.get_velocity()
                ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
                radar_detected_speed_kmh = (ego_speed + closest_speed) * 3.6
            else:
                radar_detected_speed_kmh = 0.0
            
            if self._lane_change_enabled and self.front_distance < self.vehicle_detection_distance:
                print(f"Radar detected object at {self.front_distance:.1f}m with relative speed {closest_speed:.1f}m/s")
                
            if self.node and self.radar_pub:
                msg = Range()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.header.frame_id = 'radar'
                msg.radiation_type = Range.ULTRASOUND
                msg.field_of_view = math.radians(30)
                msg.min_range = 0.0
                msg.max_range = 100.0
                msg.range = self.front_distance
                self.radar_pub.publish(msg)
                
                if int(time.time()) % 2 == 0:
                    status_msg = String()
                    status_msg.data = f"Front distance: {self.front_distance:.1f}m"
                    self.status_pub.publish(status_msg)
        except Exception as e:
            print(f"Error in radar callback: {e}")
    
    def find_adversary_vehicle(self):
        if not self.vehicle or not self.world:
            return None, 0.0, float('inf')
        
        actor_list = self.world.get_actors()
        vehicles = [actor for actor in actor_list if 'vehicle' in actor.type_id and actor.id != self.vehicle.id]
        
        if not vehicles:
            if self._lane_change_enabled:
                print("No other vehicles found in the world")
            return None, 0.0, float('inf')
            
        ego_location = self.vehicle.get_location()
        ego_waypoint = self.world.get_map().get_waypoint(ego_location, project_to_road=True)
        ego_road_id = ego_waypoint.road_id
        ego_lane_id = ego_waypoint.lane_id
        ego_transform = self.vehicle.get_transform()
        ego_forward_vector = ego_transform.get_forward_vector()
        
        closest_vehicle = None
        closest_distance = float('inf')
        
        for vehicle in vehicles:
            vehicle_location = vehicle.get_location()
            distance = ego_location.distance(vehicle_location)
            
            if distance > self.vehicle_detection_distance:
                continue
                
            vehicle_vector = carla.Vector3D(
                x=vehicle_location.x - ego_location.x,
                y=vehicle_location.y - ego_location.y,
                z=vehicle_location.z - ego_location.z
            )
            
            vehicle_vector_length = math.sqrt(vehicle_vector.x**2 + vehicle_vector.y**2 + vehicle_vector.z**2)
            if vehicle_vector_length > 0:
                vehicle_vector_normalized = carla.Vector3D(
                    x=vehicle_vector.x / vehicle_vector_length,
                    y=vehicle_vector.y / vehicle_vector_length,
                    z=vehicle_vector.z / vehicle_vector_length
                )
                
                forward_dot = ego_forward_vector.x * vehicle_vector_normalized.x + ego_forward_vector.y * vehicle_vector_normalized.y
                
                if forward_dot <= 0:
                    continue
                    
                vehicle_waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True)
                
                if vehicle_waypoint.road_id == ego_road_id and vehicle_waypoint.lane_id == ego_lane_id:
                    lateral_vector = ego_transform.get_right_vector()
                    lateral_distance = abs(lateral_vector.x * vehicle_vector.x + lateral_vector.y * vehicle_vector.y)
                    
                    if lateral_distance < self.same_lane_threshold and distance < closest_distance:
                        closest_vehicle = vehicle
                        closest_distance = distance
        
        if closest_vehicle:
            velocity = closest_vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed * 3.6
            if self._lane_change_enabled:
                print(f"Found adversary vehicle ahead at {closest_distance:.1f}m with speed {speed_kmh:.1f}km/h")
            return closest_vehicle, speed_kmh, closest_distance
        else:
            if self._lane_change_enabled:
                print("No adversary vehicle found in the same lane ahead")
            return None, 0.0, float('inf')
    
    def run_step(self, input_data, timestamp):
        if not self.vehicle:
            self.initialize()
            self._lane_change_enabled_time = timestamp + self._startup_delay
            print(f"First initialization, lane changes will be enabled after {self._startup_delay} seconds (at time {self._lane_change_enabled_time:.1f})")
        else:
            if not self._lane_change_enabled and timestamp >= self._lane_change_enabled_time:
                self._lane_change_enabled = True
                print(f"=== LANE CHANGE SYSTEM NOW ACTIVE AT TIME {timestamp:.1f} ===")
        
        if not self.vehicle.is_alive:
            print("Ego vehicle destroyed unexpectedly!")
            return carla.VehicleControl()
        
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed * 3.6
        location = self.vehicle.get_location()
        
        self.update_spectator_view()
        
        # Still call find_adversary_vehicle for logging, but it won't trigger lane change
        self.adversary_vehicle, self.adversary_speed_kmh, closest_distance = self.find_adversary_vehicle()
        
        # Fixed target speed of 40 km/h (11.11 m/s)
        target_speed = self.target_speed
        
        if int(timestamp) % 10 < 0.1:
            print(f"Time: {timestamp:.1f}, Speed: {speed_kmh:.1f}km/h, Front distance: {self.front_distance:.1f}m, Lane changes: {'ENABLED' if self._lane_change_enabled and not self._overtaking_completed else 'DISABLED'}")
        
        control = carla.VehicleControl()
        steering = self.calculate_steering()
        
        if not self._last_step_time:
            self._last_step_time = time.time()
        
        if timestamp < self._lane_change_cooldown:
            print(f"Lane change on cooldown until {self._lane_change_cooldown:.1f} (remaining: {(self._lane_change_cooldown - timestamp):.1f}s)")
        
        if self.lane_change_state != "none":
            elapsed_time = timestamp - self.lane_change_start_time
            print(f"Lane change state: {self.lane_change_state}, elapsed: {elapsed_time:.1f}s")
            
            if self.lane_change_state == "left" and elapsed_time < self.left_turn_duration:
                control.steer = steering
                control.throttle = 0.3
                control.brake = 0.0
                print(f"Executing LEFT turn with dynamic steering: steer={control.steer:.2f}")
            elif self.lane_change_state == "left" and elapsed_time >= self.left_turn_duration:
                print("LEFT turn complete. Switching to left lane waypoints")
                if self.generate_path(distance=500, lane_change='left'):
                    self.lane_change_state = "straight"
                    self.lane_change_start_time = timestamp
                    control.steer = steering
                    control.throttle = 0.3
                    control.brake = 0.0
                else:
                    print("Failed to generate left lane waypoints, aborting lane change")
                    self.lane_change_state = "none"
                    self.lane_change_start_time = None
                    control.steer = steering
            elif self.lane_change_state == "straight" and elapsed_time < self.straight_duration:
                print(f"Driving in PASSING lane: {elapsed_time:.1f}/{self.straight_duration:.1f}s")
                control.steer = steering
                control.throttle = min(0.7, (target_speed - speed) * 0.1) if speed < target_speed else 0.0
                control.brake = 0.0 if speed < target_speed else min(0.3, (speed - target_speed) * 0.1)
            elif self.lane_change_state == "straight" and elapsed_time >= self.straight_duration:
                print("Time to return to original lane after 20 seconds")
                if self.generate_path(distance=500, lane_change='right'):
                    self.lane_change_state = "right"
                    self.lane_change_start_time = timestamp
                    control.steer = steering
                    control.throttle = 0.3
                    control.brake = 0.0
                    print(f"Executing RIGHT turn with dynamic steering: steer={control.steer:.2f}")
                else:
                    print("Failed to generate right lane waypoints, staying in current lane")
                    self.lane_change_state = "none"
                    self.lane_change_start_time = None
                    control.steer = steering
            elif self.lane_change_state == "right" and elapsed_time < self.right_turn_duration:
                control.steer = steering
                control.throttle = 0.3
                control.brake = 0.0
                print(f"Executing RIGHT turn with dynamic steering: steer={control.steer:.2f}")
            else:
                print("=== LANE CHANGE MANEUVER COMPLETE ===")
                self.lane_change_state = "none"
                self.lane_change_start_time = None
                self._lane_change_cooldown = timestamp + self._cooldown_duration
                self._overtaking_completed = True
                print(f"Overtaking completed, lane changes permanently disabled")
                self.generate_path(distance=500)
                control.steer = steering
        elif (self._lane_change_enabled and not self._overtaking_completed and timestamp >= self._lane_change_cooldown and 
              self.front_distance <= self.lane_change_distance_threshold):
            # Radar-based trigger only
            print(f"[RADAR] Obstacle detected at {self.front_distance:.1f}m (within {self.lane_change_distance_threshold}m threshold)")
            if self.execute_lane_change("left", timestamp):
                control.steer = steering
                control.throttle = 0.3
                control.brake = 0.0
            else:
                control.steer = steering
        # Commented out vector-based trigger
        # elif (self._lane_change_enabled and not self._overtaking_completed and timestamp >= self._lane_change_cooldown and 
        #       self.adversary_vehicle and closest_distance <= self.lane_change_distance_threshold):
        #     print(f"[VECTOR] Obstacle detected at {closest_distance:.1f}m (within {self.lane_change_distance_threshold}m threshold)")
        #     if self.execute_lane_change("left", timestamp):
        #         control.steer = steering
        #         control.throttle = 0.3
        #         control.brake = 0.0
        #     else:
        #         control.steer = steering
        else:
            if speed < target_speed:
                control.throttle = min(0.7, (target_speed - speed) * 0.1)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(0.3, (speed - target_speed) * 0.1)
            control.steer = steering
        
        print(f"Control: throttle={control.throttle:.2f}, brake={control.brake:.2f}, steer={control.steer:.2f}, speed={speed_kmh:.2f}km/h")
        
        current_time = time.time()
        elapsed = current_time - self._last_step_time
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)
        self._last_step_time = time.time()
        
        return control
    
    def update_spectator_view(self):
        if not self.spectator or not self.vehicle:
            return
            
        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            yaw_rad = math.radians(vehicle_transform.rotation.yaw)
            
            x_offset = -math.cos(yaw_rad) * self.spectator_distance
            y_offset = -math.sin(yaw_rad) * self.spectator_distance
            
            spectator_location = carla.Location(
                x=vehicle_location.x + x_offset,
                y=vehicle_location.y + y_offset,
                z=vehicle_location.z + self.spectator_height
            )
            
            spectator_rotation = carla.Rotation(
                pitch=-20,
                yaw=vehicle_transform.rotation.yaw,
                roll=0
            )
            
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            self.spectator.set_transform(spectator_transform)
            
            if int(time.time()) % 5 == 0:
                print(f"Updated spectator view to follow vehicle")
                
        except Exception as e:
            print(f"Error updating spectator view: {e}")
    
    def destroy(self):
        print("Destroying Ros2BridgeAgent resources")
        for sensor in self.sensor_list:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        if self.node:
            self.node.destroy_node()
            rclpy.shutdown()

def main():
    return Ros2BridgeAgent()

if __name__ == '__main__':
    main()