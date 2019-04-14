from __future__ import print_function

import sys
sys.path.append("/home/zhong/Downloads/CARLA_0.8.2/PythonClient")


from lidar import run_cpp

import math
import numpy as np
import networkx as nx
import gym
import matplotlib.pyplot as plt
import argparse
import logging
import random
import time

from gym import error, spaces, utils
from gym.utils import seeding




from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


#from problem import Problem
MAX_ARRAY_LENGTH = 100
DATA = "/home/zhong/ZhongProjects/gym_routing-master/routingdataset.hdf5"
class CarlaEnv(gym.Env):
    metadata = {'render.modes': []}

    #def run_carla_client_initial(self):

    def get_args(self):
        argparser = argparse.ArgumentParser(description=__doc__)
        argparser.add_argument(
            '-v', '--verbose',
            action='store_true',
            dest='debug',
            help='print debug information')
        argparser.add_argument(
            '--host',
            metavar='H',
            default='localhost',
            help='IP of the host server (default: localhost)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '-a', '--autopilot',
            action='store_true',
            help='enable autopilot')
        argparser.add_argument(
            '-l', '--lidar',
            action='store_true',
            help='enable Lidar')
        argparser.add_argument(
            '-q', '--quality-level',
            choices=['Low', 'Epic'],
            type=lambda s: s.title(),
            default='Epic',             # 'Epic'
            help='graphics quality level, a lower level makes the simulation run considerably faster.')
        argparser.add_argument(
            '-i', '--images-to-disk',
            action='store_true',
            dest='save_images_to_disk',
            help='save images (and Lidar data if active) to disk')
        argparser.add_argument(
            '-c', '--carla-settings',
            metavar='PATH',
            dest='settings_filepath',
            default=None,
            help='Path to a "CarlaSettings.ini" file')

        return argparser.parse_args("")


    def Connect_Client(self):
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,                         # 20
            NumberOfPedestrians=40,                      # 40
            WeatherId=1,                              # random.choice([1, 3, 7, 8, 14]),
            QualityLevel=self.args.quality_level)
        settings.randomize_seeds()

        # Now we want to add a couple of cameras to the player vehicle.
        # We will collect the images produced by these cameras every
        # frame.

        # The default camera captures RGB images of the scene.
        camera0 = Camera('CameraRGB')
        # Set image resolution in pixels.
        camera0.set_image_size(800, 600)
        # Set its position relative to the car in meters.
        camera0.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera0)

        # Let's add another camera producing ground-truth depth.
        camera1 = Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        camera1.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera1)

        scene = self.client.load_settings(settings)

        player_start = 1
        #print('Starting new episode...')
        self.client.start_episode(player_start)
        #3 self.client.__exit__()


    def __init__(self):

        self.args = self.get_args()

        self.client = None
        self.client_manager = None

        self.client_manager = make_carla_client(self.args.host, self.args.port)
        self.client = self.client_manager.__enter__()
        self.Connect_Client()
        #self.client.__enter__()
        print('CarlaClient connected')

        self.velocity_front = 10
        self.dt = 1
        self.acc = 0.5
        #max_control = np.array([100,10])
        #self.action_space = spaces.Box(-max_control, max_control, dtype=np.)
        self.action_space = spaces.Discrete(10)
        self.state = None
        self.steps = 1

        high = np.array([
            200.0,125.0,90])
        low = np.array([
            -150.0,95.0,-90])


        self.observation_space = spaces.Box(low, high, dtype=np.float32)#, dtype=np.float32)
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        measurements, sensor_data = self.client.read_data()
        player_measurements = measurements.player_measurements
        pos_x=player_measurements.transform.location.x
        pos_y=player_measurements.transform.location.y
        yaw = player_measurements.transform.rotation.yaw
        speed=player_measurements.forward_speed * 3.6

        if speed > 28:
            br = (speed-28)*0.1
            thr = 0
        else:
            br = 0
            thr = (28-speed)*0.05 + 0.6

        if pos_x>60:
            br = 1
            thr = 0
            st = 0

        st = (action-5)*0.04

        self.client.send_control(
            #steer=random.uniform(-1.0, 1.0),
            steer=st,
            throttle=thr,
            brake=br,
            hand_brake=False,
            reverse=False)

        steps = self.steps


        self.state = pos_x, pos_y, yaw
        self.steps = steps + 1


        collision_vehicles = measurements.player_measurements.collision_vehicles
        intersection_offroad = measurements.player_measurements.intersection_offroad

        if abs(collision_vehicles)>1 or abs(intersection_offroad)>0:
            reward = -1
            done = True
        else:
            reward = (pos_x + 30)/120
            done = False

        if self.steps > 200:
            done = True

        if pos_x > 60 and pos_y > 108 and not done:
            reward = 1

        return np.array(self.state), reward, done, {}


    def reset(self, **kargs):
        self.state = np.array([-22.0,109.0,0.0])
        self.steps = 1

        self.Connect_Client()

        #measurements, sensor_data = self.client.read_data()
        #player_measurements = measurements.player_measurements
        # pos_x=player_measurements.transform.location.x
        # pos_y=player_measurements.transform.location.y
        # speed=player_measurements.forward_speed * 3.6
        # self.state = pos_x,pos_y


        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            #world_width = self.problem.xrange
            super(MyEnv, self).render(mode=mode)
