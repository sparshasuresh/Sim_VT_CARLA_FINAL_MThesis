#!/usr/bin/env python

import carla
import time

def main():
    client = None
    tm = None
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        
        print("Connected to CARLA. Setting spectator position and enabling Traffic Manager...")
        
        # Initialize Traffic Manager
        tm = client.get_trafficmanager(8000)  # Default TM port
        tm.set_synchronous_mode(True)  # Sync with world ticks (optional, remove if not using synchronous mode)
        print("Traffic Manager enabled on port 8000")
        
        # Get the spectator
        spectator = world.get_spectator()
        
        # Set spectator position to spawn point 299
        location = carla.Location(x=68.33110809326172, y=13.360282897949219, z=11.457353591918945 + 6.0)
        rotation = carla.Rotation(pitch=-20.0, yaw=-179.76736450195312, roll=0.0)
        transform = carla.Transform(location, rotation)
        
        # Set the spectator
        spectator.set_transform(transform)
        
        print("Spectator view set to spawn point 299")
        
        # Keep the script running to maintain spectator view
        try:
            print("Press Ctrl+C to exit")
            while True:
                world.tick()  # Required if TM is in synchronous mode
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if tm:
            tm.set_synchronous_mode(False)  # Reset TM sync mode
        if client:
            settings = world.get_settings()
            settings.synchronous_mode = False  # Reset world sync mode if used
            world.apply_settings(settings)
        print("Cleanup complete")

if __name__ == '__main__':
    main()