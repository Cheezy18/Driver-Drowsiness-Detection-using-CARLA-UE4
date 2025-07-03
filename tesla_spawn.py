import glob
import os
import sys
import argparse
import random
import time
import carla

def main():
    argparser = argparse.ArgumentParser(description="CARLA Vehicle Spawning")
    argparser.add_argument("--host", default="127.0.0.1", help="IP of the CARLA server")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port (default: 2000)")
    args = argparser.parse_args()

    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()

    # ------------------ Spawn Vehicle ------------------
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("❌ No spawn points found!")
        return

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]  # Tesla Model 3

    # Spawn vehicle at a random spawn point
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if not vehicle:
        print("❌ Vehicle spawn failed!")
        return
    vehicle.set_autopilot(True)
    print("\n✅ Vehicle spawned and running in autopilot mode!")

    # Keep running until interrupted
    try:
        while True:
            world.wait_for_tick(seconds=30.0)
            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n❌ Simulation stopped by user.")
        vehicle.destroy()
        print("✅ Vehicle removed. Exiting.")

if __name__ == "__main__":
    main()
