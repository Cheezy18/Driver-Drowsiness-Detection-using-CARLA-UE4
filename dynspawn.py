import glob
import os
import sys
import argparse
import math
import random
import time
import carla

# ------------------ Helper Function ------------------

def clamp(value, min_value, max_value):
    """Ensure a value stays within min/max limits."""
    return max(min_value, min(value, max_value))


# ------------------ Sun & Storm Classes ------------------

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        """Smoothly adjust sun position over time."""
        self._t += 0.005 * delta_seconds  # Slower transition
        self._t %= 2.0 * math.pi
        self.azimuth += 0.1 * delta_seconds  # Gradual sun movement
        self.azimuth %= 360.0
        self.altitude = clamp(80 * math.sin(self._t) - 10, -90, 90)  # Smooth sunrise/sunset

    def __str__(self):
        return '☀ Sun(alt: %.2f°, azm: %.2f°)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self):
        self.clouds = random.uniform(0, 20)  # Start with clear sky
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0
        self.target_values = self._generate_new_weather_target()

    def _generate_new_weather_target(self):
        """Generate a new weather state to transition toward over time."""
        return {
            "clouds": random.uniform(0, 100),
            "rain": random.uniform(0, 70),
            "wetness": random.uniform(0, 80),
            "puddles": random.uniform(0, 60),
            "wind": random.uniform(0, 50),
            "fog": random.uniform(0, 40),
        }

    def tick(self, delta_seconds):
        """Gradually transition towards the new weather target."""
        transition_speed = 0.005 * delta_seconds  # Slower transition
        for key in self.target_values.keys():
            setattr(self, key, clamp(
                getattr(self, key) + (self.target_values[key] - getattr(self, key)) * transition_speed,
                0, 100
            ))

        # Occasionally generate new target weather states
        if random.random() < 0.01:  # 1% chance per tick
            self.target_values = self._generate_new_weather_target()

    def __str__(self):
        return '🌧 Storm(Clouds: %d%%, Rain: %d%%, Fog: %d%%, Wind: %d%%)' % (
            self.clouds, self.rain, self.fog, self.wind)


# ------------------ Weather Control ------------------

class Weather(object):
    def __init__(self, world):
        self.weather = world.get_weather()
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm()

    def tick(self, delta_seconds):
        """Apply smooth real-time weather updates."""
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)

        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def apply(self, world):
        """Apply weather changes to the simulation."""
        world.set_weather(self.weather)

    def __str__(self):
        return '%s | %s' % (self._sun, self._storm)


# ------------------ Main Function ------------------

def main():
    argparser = argparse.ArgumentParser(description="CARLA Realistic Dynamic Weather + Vehicle Simulation")
    argparser.add_argument("--host", default="127.0.0.1", help="IP of the CARLA server")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port (default: 2000)")
    args = argparser.parse_args()

    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Initialize dynamic weather system
    weather = Weather(world)

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

    # ------------------ Simulation Loop ------------------
    try:
        while True:
            timestamp = world.wait_for_tick(seconds=30.0).timestamp
            weather.tick(timestamp.delta_seconds)
            weather.apply(world)  # Apply weather changes

            sys.stdout.write("\r🌤 Weather Update | Sun: %.2f° | Clouds: %d%% | Rain: %d%% | Fog: %d%% | Wind: %d%%   " 
                             % (weather.weather.sun_altitude_angle, weather.weather.cloudiness, 
                                weather.weather.precipitation, weather.weather.fog_density, 
                                weather.weather.wind_intensity))
            sys.stdout.flush()
            
            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n❌ Simulation stopped by user.")
        vehicle.destroy()
        print("✅ Vehicle removed. Exiting.")

if __name__ == "__main__":
    main()
