from __future__ import print_function

import sys

sys.path.append("/home/carla/Downloads/CARLAOK/PythonAPI")
sys.path.append("/home/carla/Downloads/CARLAOK/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg")

#from lidar import run_cpp

import math
import numpy as np
import networkx as nx
import gym
import matplotlib.pyplot as plt
import argparse
import logging
import random
import time
import collections
import datetime
import glob
import os
import re
import weakref
import matplotlib.pyplot as plt


from gym import error, spaces, utils
from gym.utils import seeding


import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import *
from agents.navigation.basic_agent import *
from agents.navigation.local_planner import LocalPlanner,Driving_State
from agents.tools.misc import distance_vehicle, get_speed



###########################################
class CollisionSensor(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self.collision_flag = False
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(self._on_collision)

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    def _on_collision(self, event):
        self.collision_flag = True


##########################################

#from problem import Problem
MAX_ARRAY_LENGTH = 100
DATA = "/home/zhong/ZhongProjects/gym_routing-master/routingdataset.hdf5"
class CarlaEnv(gym.Env):
    metadata = {'render.modes': []}

    #def run_carla_client_initial(self):


    def get_args(self):


        argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
        argparser.add_argument(
            '-v', '--verbose',
            action='store_true',
            dest='debug',
            help='print debug information')
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '--res',
            metavar='WIDTHxHEIGHT',
            default='1280x720',
            help='window resolution (default: 1280x720)')

        argparser.add_argument("-a", "--agent", type=str,
                               choices=["Roaming", "Basic"],
                               help="select which agent to run",
                               default="Basic")

        return argparser.parse_args("")

    def add_a_vehicle(self):

        world = self.ego_vehicle.get_world()
        # if not world.wait_for_tick(10.0):
        #     return
        blueprint_library = world.get_blueprint_library()
        r = 27
        cx = -0.4771
        cy = 0.1983

        location = self.ego_vehicle.get_location()

        bp = random.choice(blueprint_library.filter('vehicle'))
        transform = random.choice(world.get_map().get_spawn_points())

        d = (transform.location.x-location.x)*(transform.location.x-location.x)+(transform.location.y-location.y)*(transform.location.y-location.y)
        d = np.sqrt(d)

        r_new = (transform.location.x-cx)*(transform.location.x-cx)+(transform.location.y-cy)*(transform.location.y-cy)
        r_new = np.sqrt(r_new)

        # if r_new < 20:
        #     return
        for car in self.actor_list:
            if car.is_alive:
                car_location = car.get_location()
                d1 = (transform.location.x-car_location.x)*(transform.location.x-car_location.x)+(transform.location.y-car_location.y)*(transform.location.y-car_location.y)
                d1 = np.sqrt(d1)
                if d1 < d:
                    d = d1
        if d > 10 and r_new < 100:
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                self.car_num += 1
                self.actor_list.append(npc)
                npc.set_autopilot(True)


    def _carla_restart(self):

        self.args = self.get_args()
        self.args.width, self.args.height = [int(x) for x in self.args.res.split('x')]

        log_level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

        logging.info('listening to server %s:%s', self.args.host, self.args.port)

        print(__doc__)
        self.actor_list = []

        world = None
        trigger = 0
        yaw_last = -400
        client = carla.Client(self.args.host, self.args.port)
        client.set_timeout(4.0)
        Carla_state = Driving_State

        world = client.get_world()
        map = world.get_map()
        blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        spawn_points = map.get_spawn_points()
        m = spawn_points[1]
        m.location.x = m.location.x - 200
        m.location.y = m.location.y - 2.5
        spawn_point = m
        #print(spawn_point)
        self.ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.ego_vehicle)

        target_speed = 20
        self._local_planner = LocalPlanner(self.ego_vehicle, opt_dict={'target_speed' : target_speed})
        self.actor_list.append(self.ego_vehicle)
        if self.args.agent == "Roaming":
            self.agent = RoamingAgent(self.ego_vehicle)
        else:
            self.agent = BasicAgent(self.ego_vehicle)
            spawn_point = map.get_spawn_points()[0]
            self.agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))

        self.car_num = 0


    def _ego_reset(self):

        #self.ego_vehicle.destroy()
        self.collision_sensor.sensor.destroy()

        client = carla.Client(self.args.host, self.args.port)
        client.set_timeout(4.0)

        world = client.get_world()
        map = world.get_map()
        blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        spawn_points = map.get_spawn_points()
        m = spawn_points[1]
        m.location.x = m.location.x - 200
        m.location.y = m.location.y - 2.5
        spawn_point = m
        #print(spawn_point)

        self.ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.ego_vehicle)

        target_speed = 20
        self._local_planner = LocalPlanner(self.ego_vehicle, opt_dict={'target_speed' : target_speed})
        self.actor_list.append(self.ego_vehicle)
        if self.args.agent == "Roaming":
            self.agent = RoamingAgent(self.ego_vehicle)
        else:
            self.agent = BasicAgent(self.ego_vehicle)
            spawn_point = map.get_spawn_points()[0]
            self.agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))


    def __init__(self):

        self._carla_restart()
        self.action_space = spaces.Discrete(8)
        """
        action space:
        0: rule-based policy
        1: emergency brake (acc = -10)
        2: acc = 0; target to outside
        3: acc = 0; target to inside
        4: acc = 1; target to outside
        5: acc = 1; target to inside
        6: acc = -1; target to outside
        7: acc = -1; target to inside
        """
        self._restart_motivation = 0
        self.state = None
        self.steps = 1
        self.collision_times = 0
        #
        # low  = np.array([ 0,  0,  0,   0,-1,   0,-1,   0,   0])
        # high = np.array([40, 40, 60, 200, 1, 200, 1, 200, 200])

        low  = np.array([18,  0,   0, -1,  0,   0,-1,  0,   0,  0,   0,  0])
        high = np.array([40, 60, 200,  1, 36, 200, 1, 36, 200, 36, 200, 36])


        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()

    def __del__(self):
        print("destroying1")
        # client = carla.Client(self.args.host, self.args.port)
        # client.set_timeout(4.0)
        # world = client.get_world()
        #
        # actor_list = world.get_actors()
        # for actor in actor_list:
        #     if actor.is_alive:
        #         actor.destroy()

    def clean(self):
        print("destroying2")
        world = self.ego_vehicle.get_world()
        actor_list = world.get_actors()
        for actor in actor_list:
            if actor is not None:
                actor.destroy()


    def _draw_text(self,action):

        return
        world = self.ego_vehicle.get_world()
        ego_vehicle_location = self.ego_vehicle.get_location()
        pos_x = ego_vehicle_location.x
        pos_y = ego_vehicle_location.y

        point_loc = carla.Location(x = 0,y = 0, z = 20)
        text_loc = carla.Location(x = 30,y = 20, z = 4)
        text_color = carla.Color()
        if action == 0:
            text_color.r = 25
            text_color.g = 202
            text_color.b = 173
            world.debug.draw_string(text_loc, "Rule-based", draw_shadow=True, color=text_color, life_time=0.2)
            #world.debug.draw_point(point_loc, size=0.8, color=text_color, life_time=0.2)
        else:
            text_color.r = 244
            text_color.g = 96
            text_color.b = 108
            text_loc = carla.Location(x = 30,y = 16, z = 4)
            world.debug.draw_string(text_loc, "RL kicks in", draw_shadow=True, color=text_color, life_time=0.2)
            #world.debug.draw_point(point_loc, size=0.8, color=text_color, life_time=0.2)


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        steps = self.steps
        world = self.ego_vehicle.get_world()
        ego_vehicle_location = self.ego_vehicle.get_location()
        pos_x = ego_vehicle_location.x
        pos_y = ego_vehicle_location.y

        # text_loc = carla.Location(x = pos_x,y = pos_y, z = 1)
        # text_color = carla.Color()
        # text_color.r = 0
        # text_color.g = 255
        # text_color.b = 0

        #print(steps,action)
        if action == 0: ####### RL based
            while not world.wait_for_tick(10.0):
                pass
            control = self._local_planner.run_step()
            self.ego_vehicle.apply_control(control)
            self._draw_text(action)
            self._local_planner._rulebased_signal = True
        else:
            self._local_planner._rulebased_signal = False
            self._local_planner.set_target_RL(action)
            while not world.wait_for_tick(10.0):
                pass
            control = self._local_planner.run_step()
            self.ego_vehicle.apply_control(control)
            self._draw_text(action)
        #print(self._local_planner._make_decision)
        make_decision = self._local_planner._make_decision
        while not make_decision:
            #self.add_a_vehicle()
            if self.car_num < 80 or self._local_planner._empty_road:
                #print("adding vehicles: ",self.car_num)
                self.add_a_vehicle()
            while not world.wait_for_tick(10.0):
                pass
            control = self._local_planner.run_step()
            self.ego_vehicle.apply_control(control)
            self._draw_text(action)
            make_decision = self._local_planner._make_decision


        ego_vehicle_location = self.ego_vehicle.get_location()
        pos_x = ego_vehicle_location.x
        pos_y = ego_vehicle_location.y

        Carla_state = self._local_planner.get_RL_state()
        r_ego = self._local_planner._distance_to_center(self.ego_vehicle)
        self.state = r_ego, Carla_state.ego_speed,\
                     Carla_state.front_vehicle_inside_distance,\
                     Carla_state.front_vehicle_inside_direction,\
                     Carla_state.front_vehicle_inside_speed,\
                     Carla_state.front_vehicle_outside_distance,\
                     Carla_state.front_vehicle_outside_direction,\
                     Carla_state.front_vehicle_outside_speed,\
                     Carla_state.behind_vehicle_inside_distance,\
                     Carla_state.behind_vehicle_inside_speed,\
                     Carla_state.behind_vehicle_outside_distance,\
                     Carla_state.behind_vehicle_outside_speed

        """
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_speed = 0.0
        self.ego_vehicle = None

        self.front_vehicle_inside = None  ### cannot be used directly
        self.front_vehicle_inside_distance = 0.0
        self.front_vehicle_inside_direction = 0.0
        self.front_vehicle_inside_speed = 0.0

        self.front_vehicle_outside = None  ### cannot be used directly
        self.front_vehicle_outside_distance = 0.0
        self.front_vehicle_outside_direction = 0.0
        self.front_vehicle_outside_speed = 0.0

        self.behind_vehicle_inside = None  ### cannot be used directly
        self.behind_vehicle_inside_distance = 0.0
        self.behind_vehicle_inside_direction = 0.0
        self.behind_vehicle_inside_speed = 0.0

        self.behind_vehicle_outside = None  ### cannot be used directly
        self.behind_vehicle_outside_distance = 0.0
        self.behind_vehicle_outside_direction = 0.0
        self.behind_vehicle_outside_speed = 0.0
        """

        self.steps = steps + 1

        done = False
        reward = 0

        if action == 0:
            reward += 0.6
        if self.steps > 20:
            done = True

        if not self.collision_sensor.collision_flag:
            pass
            #self._restart_motivation = 0
        else:
            self.collision_times += 1
            print("total_collision:",self.collision_times)
            self.collision_sensor.collision_flag = False
            reward = -1
            #self._restart_motivation += 1

        if get_speed(self.ego_vehicle) < 1:
            self._restart_motivation += 1
        else:
            self._restart_motivation = 0

        ego_vehicle_location = self.ego_vehicle.get_location()
        pos_x = ego_vehicle_location.x
        pos_y = ego_vehicle_location.y

        if abs(pos_x)>36 or abs(pos_y)>36:
            done = True
            self._ask_for_restart()

        if self._restart_motivation > 30:
            done = True
            self._ask_for_restart()


        return np.array(self.state), reward, done, {}

    def _ask_for_restart(self):
        ego_vehicle_location = self.ego_vehicle.get_location()
        pos_x = ego_vehicle_location.x
        pos_y = ego_vehicle_location.y
        if abs(pos_x)>36 or abs(pos_y)>36:
            #print("restarting1")
            self._ego_reset()
            self._restart_motivation = 0
            return

        if self._restart_motivation > 30:
            #print("restarting2")
            self._ego_reset()
            self._restart_motivation = 0
            return

    def reset(self, **kargs):
        world = self.ego_vehicle.get_world()

        while self.collision_sensor.collision_flag:
            self.collision_sensor.collision_flag = False
            make_decision = self._local_planner._make_decision
            while not make_decision:
                self.add_a_vehicle()
                world.wait_for_tick(10.0)
                control = self._local_planner.run_step()
                self.ego_vehicle.apply_control(control)
                make_decision = self._local_planner._make_decision
        #########
        # Here we need to find a stable state for a new set
        #
        #######

        ego_vehicle_location = self.ego_vehicle.get_location()
        pos_x = ego_vehicle_location.x
        pos_y = ego_vehicle_location.y
        #print("------------Fake Reset RL--------------------")
        #self.state = pos_x, pos_y, 0
        Carla_state = self._local_planner.get_RL_state()
        r_ego = self._local_planner._distance_to_center(self.ego_vehicle)
        self.state = r_ego, Carla_state.ego_speed,\
                     Carla_state.front_vehicle_inside_distance,\
                     Carla_state.front_vehicle_inside_direction,\
                     Carla_state.front_vehicle_inside_speed,\
                     Carla_state.front_vehicle_outside_distance,\
                     Carla_state.front_vehicle_outside_direction,\
                     Carla_state.front_vehicle_outside_speed,\
                     Carla_state.behind_vehicle_inside_distance,\
                     Carla_state.behind_vehicle_inside_speed,\
                     Carla_state.behind_vehicle_outside_distance,\
                     Carla_state.behind_vehicle_outside_speed
        ## Carla_state.ego_x,Carla_state.ego_y
        self.steps = 1

        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            #world_width = self.problem.xrange
            super(MyEnv, self).render(mode=mode)
