import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    spawn_points = world.get_map().get_spawn_points()
    
    # Visualize all spawn points
    for i, point in enumerate(spawn_points):
        world.debug.draw_string(point.location, f'Spawn {i}', color=carla.Color(255,0,0), life_time=60.0)
        world.debug.draw_point(point.location, size=0.1, color=carla.Color(255,0,0), life_time=60.0)
        
        forward = point.get_forward_vector()
        end_loc = point.location + carla.Location(x=forward.x*2, y=forward.y*2, z=forward.z*2)
        world.debug.draw_arrow(point.location, end_loc, thickness=0.1, arrow_size=0.1, color=carla.Color(0,255,0), life_time=60.0)
    
    # Specifically print coordinates for points 308 and 161
    target_indices = [235, 287]
    for idx in target_indices:
        if idx < len(spawn_points):
            point = spawn_points[idx]
            print(f"\nCoordinates for Spawn Point {idx}:")
            print(f"Location: x={point.location.x}, y={point.location.y}, z={point.location.z}")
            print(f"Rotation: pitch={point.rotation.pitch}, yaw={point.rotation.yaw}, roll={point.rotation.roll}")
            
            # Highlight these specific points with blue color (larger)
            world.debug.draw_point(point.location, size=0.3, color=carla.Color(0,0,255), life_time=60.0)
        else:
            print(f"Spawn point {idx} doesn't exist. Only {len(spawn_points)} spawn points available.")
    
    print(f"\nVisualized {len(spawn_points)} spawn points. They will remain visible for 60 seconds.")
    time.sleep(60)

if __name__ == '__main__':
    main()
