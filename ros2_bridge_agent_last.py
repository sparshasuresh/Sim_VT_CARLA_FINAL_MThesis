#!/usr/bin/env python3
 
import rclpy

import carla

from srunner.autoagents.autonomous_agent import AutonomousAgent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from rclpy.executors import MultiThreadedExecutor

import time
 
class Ros2BridgeAgent(AutonomousAgent):

    def __init__(self, path_to_conf_file=None):

        super().__init__(path_to_conf_file)

        print("Ros2BridgeAgent initialized")

        self.vehicle = None

        self.world = None

        self.radar_node = None

        self.decider_node = None

        self.controller_node = None

        self.initialized = False

        self.executor = None

        self._last_step_time = None

    def setup(self, path_to_conf_file):

        try:

            rclpy.init(args=None)

            self.world = CarlaDataProvider.get_world()

            if not self.world:

                print("Failed to get world")

                return

            print("World retrieved, waiting for vehicle in run_step")

        except Exception as e:

            print(f"Setup error: {e}")

    def sensors(self):

        return []

    def run_step(self, input_data, timestamp):

        if not self.initialized:

            max_attempts = 10

            for attempt in range(max_attempts):

                self.vehicle = CarlaDataProvider.get_hero_actor()

                if self.vehicle:

                    print(f"Found ego vehicle with ID {self.vehicle.id}")

                    self.vehicle.set_simulate_physics(True)

                    # Set initial forward orientation

                    transform = self.vehicle.get_transform()

                    transform.rotation.yaw = 0.0  # Face forward along +X

                    self.vehicle.set_transform(transform)

                    break

                print(f"Attempt {attempt + 1}/{max_attempts}: Still waiting for ego vehicle...")

                time.sleep(0.1)

            if not self.vehicle:

                print("Failed to find ego vehicle after retries, returning dummy control")

                return carla.VehicleControl()
 
            try:

                from radar_publisher import RadarPublisher

                from overtaking_decider import OvertakingDecider

                from controller import Controller

                self.radar_node = RadarPublisher(self.vehicle, self.world)

                self.decider_node = OvertakingDecider(self.vehicle, self.world)

                self.controller_node = Controller(self.vehicle)

                self.executor = MultiThreadedExecutor(num_threads=3)

                self.executor.add_node(self.radar_node)

                self.executor.add_node(self.decider_node)

                self.executor.add_node(self.controller_node)

                import threading

                self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)

                self.spin_thread.start()

                print("All ROS2 nodes initialized and spinning with MultiThreadedExecutor")

                self.initialized = True

            except Exception as e:

                print(f"Node initialization error: {e}")

                return carla.VehicleControl()

        if not self._last_step_time:

            self._last_step_time = time.time()

        current_time = time.time()

        elapsed = current_time - self._last_step_time

        if elapsed < 0.05:

            time.sleep(0.05 - elapsed)

        self._last_step_time = current_time

        return carla.VehicleControl()

    def destroy(self):

        print("Destroying Ros2BridgeAgent resources")

        if self.radar_node:

            self.radar_node.destroy()

        if self.executor:

            self.executor.shutdown()

        rclpy.shutdown()
 
def main():

    return Ros2BridgeAgent()
 
if __name__ == '__main__':

    main()
 