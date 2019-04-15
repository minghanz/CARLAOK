#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random
import math
import numpy as np

import carla

import os

#from agents.navigation.controller import VehiclePIDController
from agents.navigation.controller import VehiclePIDController_Nowaypoint
from agents.tools.misc import distance_vehicle, draw_waypoints
from agents.tools.misc import distance_vehicle, get_speed
from agents.tools.clock import WorldClock
from agents.tools.subscriber import lidar as ulidar
from agents.tools.subscriber import camera as ucamera
from agents.visualization.dashpage import data_append

class PercpVeh():
    def __init__(self):
        # self.location = None #carla.Location(x=0, y=0, z=0)
        # self.speed = None
        self.last_location = None #carla.Location(x=0, y=0, z=0)
        self._cx = -0.4771
        self._cy = 0.1983

        self.id = -1
        self.time = None
        self.state = None
        self.sigma = None
        self.mot_noise = np.diag([1, 1, 3, 3])
        self.obs_noise = np.diag([0.5, 0.5])

    def filter(self, x, y, time):
        if self.state is None:
            self.time = time
            self.state = np.array([x, y, 5, 5])[:,np.newaxis]
            self.sigma = np.diag([1,1,50,50])
        else:
            self.last_location = self.location
            dt = time - self.time
            print(dt)
            self.time = time
            A = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            C = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            self.state = A.dot(self.state)
            self.sigma = A.dot(self.sigma).dot(A.T) + self.mot_noise

            z = np.array([[x], [y]]) - C.dot(self.state)
            S = self.obs_noise + C.dot(self.sigma).dot(C.T)
            K = self.sigma.dot(C.T).dot(np.linalg.inv(S))
            self.state += K.dot(z)
            self.sigma = (np.diag([1]*4) - K.dot(C)).dot(self.sigma)

    @property
    def location(self):
        return carla.Location(x=float(self.state[0]), y=float(self.state[1]), z=1)

    @property
    def speed(self):
        return 3.6 * math.sqrt(self.state[2]**2 + self.state[3]**2)

    @property
    def cosangle(self):

        v_vec = self.state[2:]
        w_vec = np.array([self._cx - self.state[0], self._cy - self.state[1]])

        cosangle = np.clip(np.dot(w_vec.T, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0)
        return float(cosangle)

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4

class Driving_State(object):
    """
    AV Driving_State
    """
    def __init__(self):

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

        self.front_vehicle_inside_percp = None
        self.front_vehicle_outside_percp = None
        self.behind_vehicle_inside_percp = None
        self.behind_vehicle_outside_percp = None

class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict={}):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        self._clock = WorldClock(self._vehicle.get_world())

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self._target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None


        self._dt_real = 0.05
        self._inroundabout = False
        self._inroundabout_inside = False
        self._inroundabout_lane = 0.0
        self._r1 = 19.8
        self._r2 = 23.5
        self._last_speed = 0
        self._acceleration = 0.1
        self._r_target = 19.8
        self._cx = -0.4771
        self._cy = 0.1983
        self._necessarity = 0   ############## necessarity for lane change
        self._decision_dt = 0.75  ########### time_step_for decision
        self._decision_time_accumulation = 0.0
        self._lane_change_time = 0.75
        self._decision_step = 0  ######### the decision steps
        self._make_decision = False
        self._empty_road = False
        self._RL_state = Driving_State
        self._aggressive = True
        self._rulebased_signal = False
        self._total_running_time = 0.0


        self._perception_enable = True
        self._marker_color_lidar = carla.Color()
        self._marker_color_camera = carla.Color()
        self._marker_color_filter = carla.Color()
        self._marker_color_filter.r = 255
        self._marker_color_filter.g = 255
        self._marker_color_filter.b = 255
        

        self._percp_veh_list = []

        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=30)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._waypoints_queue_decision = deque(maxlen=30)

        # initializing controller
        self.init_controller(opt_dict)

    def __del__(self):
        self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6  # 0.5 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1,#1.95,
            'K_D': 0.0001,#0.01,
            'K_I': 8.4,#1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 2,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if 'dt' in opt_dict:
            self._dt = opt_dict['dt']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'sampling_radius' in opt_dict:
            self._sampling_radius = self._target_speed * \
                opt_dict['sampling_radius'] / 3.6
        if 'lateral_control_dict' in opt_dict:
            args_lateral_dict = opt_dict['lateral_control_dict']
        if 'longitudinal_control_dict' in opt_dict:
            args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())
        # self._vehicle_controller = VehiclePIDController(self._vehicle,
        #                                                 args_lateral=args_lateral_dict,
        #                                                 args_longitudinal=args_longitudinal_dict)
        self._vehicle_controller = VehiclePIDController_Nowaypoint(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append( (self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=80)


    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        r = self._r1
        cx = -0.4771
        cy = 0.1983

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            AV_x = last_waypoint.transform.location.x
            AV_y = last_waypoint.transform.location.y
            e = abs((cx-AV_x)*(cx-AV_x)+(cy-AV_y)*(cy-AV_y)-r*r)
            if e<40:
                self._inroundabout = True
            #print(self._inroundabout)
            if not self._inroundabout:
                next_waypoints = list(last_waypoint.next(self._sampling_radius))
                if len(next_waypoints) == 1:
                    # only one option available ==> lanefollowing
                    next_waypoint = next_waypoints[0]
                    road_option = RoadOption.LANEFOLLOW
                else:
                    # random choice between the possible options
                    road_options_list = retrieve_options(
                        next_waypoints, last_waypoint)
                    road_option = random.choice(road_options_list)
                    road_option = road_options_list[1]
                    #print(" ".join(next_waypoints[road_options_list.index(op)] for op in road_options_list))
                    #for op in road_options_list:
                    #    print(next_waypoints[road_options_list.index(op)])
                    #print(" ",last_waypoint)
                    next_waypoint = next_waypoints[road_options_list.index(
                        road_option)]
            else:
                dtheta = np.pi/180*15
                theta_last = np.arctan2(AV_y-cy,AV_x-cx)
                theta_next = theta_last-dtheta

                next_location = last_waypoint.transform.location
                next_location.x = cx+r*np.cos(theta_next)
                next_location.y = cy+r*np.sin(theta_next)
                next_waypoint = self._map.get_waypoint(next_location)
                road_option = RoadOption.LANEFOLLOW

            self._waypoints_queue.append((next_waypoint, road_option))


    def set_global_plan(self, current_plan):
        return
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True


    def _distance_to_center(self, vehicle):
        vehicle_location = vehicle.get_location()
        d = (vehicle_location.x-self._cx)*(vehicle_location.x-self._cx)+(vehicle_location.y-self._cy)*(vehicle_location.y-self._cy)
        d = np.sqrt(d)
        return d

    def _direction_to_center(self, vehicle):

        vehicle_transform = vehicle.get_transform()
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
        # v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        # w_vec = np.array([self._cx-
        #                   v_begin.x, self._cy -
        #                   v_begin.y, 0.0])

        # cosangle = np.clip(np.dot(w_vec, v_vec) /
        #                  (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0)
        cosangle = self.calc_angle(v_begin, v_end)

        return cosangle

    def calc_angle(self, v_begin, v_end):
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([self._cx - v_begin.x,
                          self._cy - v_begin.y,
                          0.0])

        cosangle = np.clip(np.dot(w_vec, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0)
        return cosangle

    # def _direction_to_cepter_percp_veh(self, vehicle):
    #     v_begin = vehicle.last_location
    #     v_end = vehicle.location
    #     cosangle = self.calc_angle(v_begin, v_end)

    #     return cosangle

    def _vehicle_in_lane_change(self,vehicle):

        vehicle_transform = vehicle.get_transform()
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([self._cx-
                          v_begin.x, self._cy -
                          v_begin.y, 0.0])

        cosangle = np.clip(np.dot(w_vec, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0)

        if abs(cosangle)<0.2:
            return False

        return True

    def _update_decision_state(self):
        self._inroundabout_lane
        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_location = self._vehicle.get_location()
        ego_r = self._distance_to_center(self._vehicle)
        current_speed = get_speed(self._vehicle)

        if ego_r > self._r2+4 or not self._inroundabout_inside:
            self._inroundabout_lane = 0.0
            return

        if self._vehicle_in_lane_change(self._vehicle) and current_speed>10:
            self._inroundabout_lane = 1.5
            return

        if ego_r < self._r1 + 1:
            self._inroundabout_lane = 1.0
            return

        if ego_r > self._r2 + 1:
            self._inroundabout_lane = 2.0
            return

        self._inroundabout_lane = 1.5
        return



    def _IDM_desired_acc(self,front_vehicle,dis_to_front,front_speed,front_direction):

        ignore_front_vehicle = False
        if self._aggressive:
            g0 = 7
            T = 0.6
            if dis_to_front > 50:
                ignore_front_vehicle = True
        else:
            g0 = 12
            T = 1.6
        v = get_speed(self._vehicle)
        if v<10:
            a = 2.73 + (10-v)/10*2
        else:
            a = 2.73

        if self._empty_road:
            v0 = 50/3.6
        else:
            v0 = 36/3.6

        b = 1.65
        delta = 4#16

        v = v/3.6
        if dis_to_front > 200 or ignore_front_vehicle:
            dv = 0
            g = dis_to_front
            g1 = 0
        else:
            v_f = front_speed/3.6
            cosangle = front_direction
            v_f = np.sqrt(v_f*v_f*(1-cosangle*cosangle))

            dv = v-v_f
            g = dis_to_front
            g1 = g0+T*v+v*dv/(2*np.sqrt(a*b))

        return a*(1-pow(v/v0,delta)-(g1/g)*((g1/g)))


    def _IDM_desired_speed(self,front_vehicle,dis_to_front,front_speed,front_direction,dt):

        v = get_speed(self._vehicle)/3.6

        desired_acc = self._IDM_desired_acc(front_vehicle,dis_to_front,front_speed,front_direction)

        return (v + desired_acc*dt)*3.6


    def _Lane_change_decision(self,
                                front_vehicle_inside,
                                front_vehicle_outside,
                                behind_vehicle_inside,
                                behind_vehicle_outside):


        r_target = self._r_target
        ego_vehicle_location = self._vehicle.get_location()
        ego_r = self._distance_to_center(self._vehicle)
        current_speed = get_speed(self._vehicle)

        front_vehicle_inside_distance = self._RL_state.front_vehicle_inside_distance
        front_vehicle_inside_speed = self._RL_state.front_vehicle_inside_speed
        front_vehicle_inside_direction = self._RL_state.front_vehicle_inside_direction

        front_vehicle_outside_distance = self._RL_state.front_vehicle_outside_distance
        front_vehicle_outside_speed = self._RL_state.front_vehicle_outside_speed
        front_vehicle_outside_direction = self._RL_state.front_vehicle_outside_direction

        behind_vehicle_inside_distance = self._RL_state.behind_vehicle_inside_distance
        behind_vehicle_outside_distance = self._RL_state.behind_vehicle_outside_distance

        IDM_v_inside_decision = self._IDM_desired_speed(front_vehicle_inside,front_vehicle_inside_distance,front_vehicle_inside_speed,front_vehicle_inside_direction,1.5)
        IDM_v_outside_decision = self._IDM_desired_speed(front_vehicle_outside,front_vehicle_outside_distance,front_vehicle_outside_speed,front_vehicle_outside_direction,1.5)
        change_to_outside = False
        change_to_inside = False
        ########### Main for inside
        if self._aggressive:
            speed_threshold = 4
            necessarity_threshold = 2
        else:
            speed_threshold = 8
            necessarity_threshold = 4

        if self._inroundabout_lane < 2.0:
            if behind_vehicle_outside_distance > 20:
                change_to_outside = True
            else:
                behind_speed = self._RL_state.behind_vehicle_outside_speed
                #behind_speed = get_speed(behind_vehicle_outside)
                ttc = behind_vehicle_outside_distance/(behind_speed-current_speed)
                if ttc > 3 and behind_vehicle_outside_distance>10:
                    change_to_outside = True
                if ttc < 0 and behind_vehicle_outside_distance>10:
                    change_to_outside = True


            if  IDM_v_outside_decision > IDM_v_inside_decision + speed_threshold and front_vehicle_inside_distance < 100:
                self._necessarity +=1
                #print(self._necessarity)
            else:
                self._necessarity = 0
            if  IDM_v_outside_decision > IDM_v_inside_decision + speed_threshold and front_vehicle_inside_distance < 100 and change_to_outside and self._necessarity > necessarity_threshold:
                #r_target = self._r2
                return self._r2

        if self._inroundabout_lane > 1.0:
            if behind_vehicle_inside_distance > 20:
                change_to_inside = True
            else:
                behind_speed = self._RL_state.behind_vehicle_inside_speed
                #behind_speed = get_speed(behind_vehicle_inside)
                ttc = behind_vehicle_inside_distance/(behind_speed-current_speed)
                if ttc > 3 and behind_vehicle_inside_distance>10 and front_vehicle_inside_distance>15:
                    change_to_inside = True
                if ttc < 0 and behind_vehicle_inside_distance>10 and front_vehicle_inside_distance>15:
                    change_to_inside = True
            if change_to_inside and IDM_v_outside_decision < IDM_v_inside_decision + speed_threshold:
                #r_target = self._r1
                return self._r1

        return r_target
        ########### Main for outside
        # if self._inroundabout_lane < 2.0:
        #     if behind_vehicle_outside_distance > 20:
        #         change_to_outside = True
        #     else:
        #         behind_speed = self._RL_state.behind_vehicle_outside_speed
        #         behind_speed = get_speed(behind_vehicle_outside)
        #         ttc = behind_vehicle_outside_distance/(behind_speed-current_speed)
        #         if ttc > 3 and behind_vehicle_outside_distance>5:
        #             change_to_outside = True
        #         if ttc < 0 and behind_vehicle_outside_distance>5:
        #             change_to_outside = True
        #
        #     if change_to_outside and IDM_v_inside_decision < IDM_v_outside_decision + 15:
        #         self._r_target = self._r2
        #         return
        #
        # if self._inroundabout_lane > 1.0:
        #     if behind_vehicle_inside_distance > 20:
        #         change_to_inside = True
        #     else:
        #         behind_speed = get_speed(behind_vehicle_inside)
        #         ttc = behind_vehicle_inside_distance/(behind_speed-current_speed)
        #         if ttc > 3 and behind_vehicle_inside_distance>5:
        #             change_to_inside = True
        #         if ttc < 0 and behind_vehicle_inside_distance>5:
        #             change_to_inside = True
        #
        #     if  IDM_v_inside_decision > IDM_v_outside_decision + 10 and front_vehicle_outside_distance < 30:
        #         self._necessarity +=1
        #     else:
        #         self._necessarity = 0
        #
        #     if  IDM_v_inside_decision > IDM_v_outside_decision + 10 and front_vehicle_outside_distance < 30 and change_to_inside and self._necessarity > 4:
        #         self._r_target = self._r1
        #         return


        ##############################################################


    def _longitudinal_decision(self,front_vehicle_inside,front_vehicle_outside):
        #self._acceleration

        front_vehicle_inside_distance = self._RL_state.front_vehicle_inside_distance
        front_vehicle_inside_speed = self._RL_state.front_vehicle_inside_speed
        front_vehicle_inside_direction = self._RL_state.front_vehicle_inside_direction
        front_vehicle_outside_distance = self._RL_state.front_vehicle_outside_distance
        front_vehicle_outside_speed = self._RL_state.front_vehicle_outside_speed
        front_vehicle_outside_direction = self._RL_state.front_vehicle_outside_direction

        acc = self._acceleration
        current_speed = get_speed(self._vehicle)
        self._last_speed = current_speed

        if self._inroundabout_lane == 1.5 and current_speed > 5 and (front_vehicle_inside_distance < 20 or front_vehicle_outside_distance <20):
            return 0

        IDM_v_inside = self._IDM_desired_speed(front_vehicle_inside,front_vehicle_inside_distance,front_vehicle_inside_speed,front_vehicle_inside_direction,self._decision_dt)
        IDM_v_outside = self._IDM_desired_speed(front_vehicle_outside,front_vehicle_outside_distance,front_vehicle_outside_speed,front_vehicle_outside_direction,self._decision_dt)
        IDM_v = 20

        if self._inroundabout_lane == 1.0 and self._r_target == self._r1:
            IDM_v = IDM_v_inside

        if self._inroundabout_lane == 2.0 and self._r_target == self._r2:
            IDM_v = IDM_v_outside

        if self._inroundabout_lane > 1.5 and self._r_target == self._r1:
            if front_vehicle_outside_distance < 30:
                v_front = front_vehicle_outside_speed
                v_upper = ((front_vehicle_outside_distance-8)/(self._lane_change_time+0.4))*3.6 + v_front
            else:
                v_upper = 30
            IDM_v = min(IDM_v_inside,v_upper)

        if self._inroundabout_lane < 1.5 and self._r_target == self._r2:
            if front_vehicle_inside_distance < 30:
                v_front = front_vehicle_inside_speed
                v_upper = ((front_vehicle_inside_distance-8)/(self._lane_change_time+0.4))*3.6 + v_front
            else:
                v_upper = 30
            IDM_v = min(IDM_v_outside,v_upper)

        if self._inroundabout_lane == 1.5:
            IDM_v = min(IDM_v_inside,IDM_v_outside)

        #decision_total = int(self._decision_dt/self._dt)
        acc = (IDM_v-current_speed)/self._decision_dt

        return acc

        # if IDM_v < 15 :
        #     if front_vehicle_inside_distance > 30 and self._inroundabout_lane == 1.0:
        #         IDM_v = 15
        #     if front_vehicle_outside_distance > 30 and self._inroundabout_lane == 2.0:
        #         IDM_v = 15


    def _environment_perception(self):
        """
        Update the environment information from sensor or true value. Save in RL_state
        """
        ego_vehicle_location = self._vehicle.get_location()
        world_decision = self._vehicle.get_world()
        actor_list = world_decision.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        roundabout_vehicle_list = []

#############################################################
#############################################################
################     SENSOR AREA       ######################
#############################################################
#############################################################

        if self._perception_enable:
            dist_thres = 6
            self_rt = self._vehicle.get_transform()
            lidar_detections = ulidar.get_global(self_rt)
            camera_detections, local_cam_detections = ucamera.get_global(self_rt)
            sensor_detections = np.vstack((lidar_detections, camera_detections))
            nsensor = sensor_detections.shape[0] if sensor_detections.size > 0 else 0
            ideal_detections = []
            ideal_car_obj = []
            for car in vehicle_list:
                car_location = car.get_location()
                # if car.is_alive and car_location.x > 1e-4 and car_location.y > 1e-4:
                if car.is_alive and car_location.x**2 + car_location.y**2 < 1000:
                    ideal_detections.append((car_location.x, car_location.y))
                    ideal_car_obj.append(car)

            # print(local_cam_detections)
            # print('------------------------')
            # print(sensor_detections)
            # print('------------------------')
            # print(ideal_detections)
            # print('--------------------------------------------------------')
            all_matched = set()
            dist_list = []
            matched_car = []
            if nsensor * len(ideal_detections) > 0:
                for iid, icar in enumerate(ideal_detections):
                    d = np.linalg.norm(sensor_detections[:,1:] - icar, axis=1)
                    mind = np.min(d)
                    dist_list.append(mind)
                    if mind > dist_thres: continue
                    sid = np.argmin(d)
                    all_matched.add(sid)
                    exist_in_list = False
                    for pv in self._percp_veh_list:
                        if ideal_car_obj[iid].id == pv.id:
                            exist_in_list = True
                            percp_veh = pv
                            percp_veh.filter(sensor_detections[sid, 1], sensor_detections[sid, 2], self._clock.time)
                            # percp_veh.last_location = percp_veh.location
                            # percp_veh.location = carla.Location(sensor_detections[sid, 1], y=sensor_detections[sid, 2], z = 0)# ideal_car_obj[iid].get_location()
                            
                            # dt = sensor_detections[sid, 0] - percp_veh.time
                            # percp_veh.time = sensor_detections[sid, 0]

                            # v_x = (percp_veh.location.x - percp_veh.last_location.x ) / self._dt_real
                            # v_y = (percp_veh.location.y - percp_veh.last_location.y ) / self._dt_real

                            # percp_veh.speed = (percp_veh.speed + 3.6 * np.sqrt(v_x**2 + v_y**2) ) /2
                            
                            # percp_veh.cosangle = self._direction_to_cepter_percp_veh(percp_veh)
                            
                            print(pv.id, "speed: ", percp_veh.speed)
                            print(ideal_car_obj[iid].id, "speed: ", get_speed(ideal_car_obj[iid]) )
                            

                    #print("exist_in_list", exist_in_list)
                    if exist_in_list == False:
                        percp_veh = PercpVeh()
                        percp_veh.id = ideal_car_obj[iid].id
                        percp_veh.filter(sensor_detections[sid, 1], sensor_detections[sid, 2], self._clock.time)
                        # percp_veh.location = carla.Location(sensor_detections[sid, 1], y=sensor_detections[sid, 2], z = 0)# ideal_car_obj[iid].get_location()
                        # percp_veh.speed = 5
                        # percp_veh.cosangle = 0
                        self._percp_veh_list.append(percp_veh)

                    # matched_car.append((ideal_car_obj[iid], sid))
                    matched_car.append((ideal_car_obj[iid], percp_veh))

            #print('---------------------------------------')

            # print(dist_list)
            # all_matched = range(len(sensor_detections))

            self._marker_color_lidar.r = (self._marker_color_lidar.r + 20)%255
            self._marker_color_lidar.g = (self._marker_color_lidar.b + 40)%255
            self._marker_color_camera.r = (self._marker_color_camera.r + 20)%255
            self._marker_color_camera.b = (self._marker_color_camera.g + 60)%255
            for lid in all_matched: # all_matched:
                d,lx,ly = sensor_detections[lid]
                if lid < len(lidar_detections):
                    world_decision.debug.draw_point(carla.Location(x=lx,y=ly,z=1),
                        size=0.06, color=self._marker_color_lidar, life_time=0.1)
                else:
                    pass
                    world_decision.debug.draw_point(carla.Location(x=lx,y=ly,z=1),
                        size=0.06, color=self._marker_color_camera, life_time=0.1)
            for _, pv in matched_car:
                lx, ly = pv.location.x, pv.location.y
                world_decision.debug.draw_point(carla.Location(x=lx,y=ly,z=1),
                    size=0.1, color=self._marker_color_filter, life_time=0.2)
                

            # for cur_detect in ideal_detections: # all_matched:
            #     world_decision.debug.draw_point(carla.Location(x=cur_detect[0],y=cur_detect[1],z=1),
            #     size=0.1, color=self._marker_color_camera, life_time=0.2)


#############################################################
#############################################################
################     END SENSOR AREA      ###################
#############################################################
#############################################################

        cx = self._cx
        cy = self._cy
        front_vehicle_inside = None
        front_vehicle_outside = None
        front_vehicle_inside_distance = 1000
        front_vehicle_inside_direction = 0
        front_vehicle_inside_speed = 50.0
        front_vehicle_outside_distance = 1000
        front_vehicle_outside_direction = 0
        front_vehicle_outside_speed = 50.0

        behind_vehicle_inside = None
        behind_vehicle_outside = None
        behind_vehicle_inside_distance = 1000
        behind_vehicle_inside_speed = 0.0
        behind_vehicle_outside_distance = 1000
        behind_vehicle_outside_speed = 0.0

        front_vehicle_inside_percp = None
        front_vehicle_outside_percp = None
        behind_vehicle_inside_percp = None
        behind_vehicle_outside_percp = None

        if self._perception_enable:
            for target_vehicle, percp_veh in matched_car: #for target_vehicle in vehicle_list:
                if target_vehicle.id == self._vehicle.id:
                    continue
                if target_vehicle is None:
                    continue
                # loc = target_vehicle.get_location()
                loc = percp_veh.location # carla.Location(x=percp_veh.location.x, y=percp_veh.location.y,  z = 0)
                distance_to_roundabout = self._distance_to_center(target_vehicle)
                if distance_to_roundabout > 120 or loc.x < -50:
                    target_vehicle.destroy()
                    continue
                theta_target_vehicle = np.arctan2(loc.y-cy,loc.x-cx)
                theta_ego_vehicle = np.arctan2(ego_vehicle_location.y-cy,ego_vehicle_location.x-cx)
                target_vehicle_speed = percp_veh.speed # get_speed(target_vehicle)
                prediction_threshold = 5
                dtheta = (-theta_target_vehicle + theta_ego_vehicle) % (2*np.pi)
                cosangle_real = self._direction_to_center(target_vehicle)
                cosangle = percp_veh.cosangle
                print("----------", cosangle_real, cosangle,"----------")

                if distance_to_roundabout < self._r1 + 2:
                    distance = self._r1*(dtheta)
                    if distance < front_vehicle_inside_distance:
                        front_vehicle_inside_distance = distance
                        front_vehicle_inside = target_vehicle
                        front_vehicle_inside_speed = target_vehicle_speed
                        front_vehicle_inside_direction = cosangle

                        front_vehicle_inside_percp = percp_veh

                    behind_distance = self._r1*(2*np.pi-dtheta)
                    if behind_distance < behind_vehicle_inside_distance:
                        behind_vehicle_inside_distance = behind_distance
                        behind_vehicle_inside = target_vehicle
                        behind_vehicle_inside_speed = target_vehicle_speed
                        
                        behind_vehicle_inside_percp = percp_veh

                    if cosangle < -0.1 and target_vehicle_speed > prediction_threshold:
                        distance = self._r2*(dtheta)
                        if distance < front_vehicle_outside_distance:
                            front_vehicle_outside_distance = distance
                            front_vehicle_outside = target_vehicle
                            front_vehicle_outside_speed = target_vehicle_speed
                            front_vehicle_outside_direction = cosangle

                            front_vehicle_outside_percp = percp_veh

                if distance_to_roundabout >= self._r1 + 1 and distance_to_roundabout < self._r2 + 4:
                    distance = self._r2*(dtheta)
                    if distance < front_vehicle_outside_distance:
                        front_vehicle_outside_distance = distance
                        front_vehicle_outside = target_vehicle
                        front_vehicle_outside_speed = target_vehicle_speed
                        front_vehicle_outside_direction = cosangle

                        front_vehicle_outside_percp = percp_veh

                    behind_distance = self._r2*(2*np.pi-dtheta)
                    if behind_distance < behind_vehicle_outside_distance:
                        behind_vehicle_outside_distance = behind_distance
                        behind_vehicle_outside = target_vehicle
                        behind_vehicle_outside_speed = target_vehicle_speed

                        behind_vehicle_outside_percp = percp_veh

                if distance_to_roundabout >= self._r1 + 1 and distance_to_roundabout < self._r2 + 13:
                    if cosangle > 0.2 and target_vehicle_speed > prediction_threshold:
                        distance = self._r1*(dtheta)
                        if distance < front_vehicle_inside_distance:
                            front_vehicle_inside_distance = distance
                            front_vehicle_inside = target_vehicle
                            front_vehicle_inside_speed = target_vehicle_speed
                            front_vehicle_inside_direction = cosangle

                            front_vehicle_inside_percp = percp_veh


                if distance_to_roundabout >= self._r2 + 1 and distance_to_roundabout < self._r2 + 13:
                    if cosangle > 0.2 and target_vehicle_speed > prediction_threshold:
                        distance = self._r2*(dtheta)
                        if distance < front_vehicle_outside_distance:
                            front_vehicle_outside_distance = distance
                            front_vehicle_outside = target_vehicle
                            front_vehicle_outside_speed = target_vehicle_speed
                            front_vehicle_outside_direction = cosangle

                            front_vehicle_outside_percp = percp_veh
        
        else:
            for target_vehicle in vehicle_list:
                if target_vehicle.id == self._vehicle.id:
                    continue
                if target_vehicle is None:
                    continue
                loc = target_vehicle.get_location()
                #loc = percp_veh.location # carla.Location(x=percp_veh.location.x, y=percp_veh.location.y,  z = 0)
                distance_to_roundabout = self._distance_to_center(target_vehicle)
                if distance_to_roundabout > 120 or loc.x < -50:
                    target_vehicle.destroy()
                    continue
                theta_target_vehicle = np.arctan2(loc.y-cy,loc.x-cx)
                theta_ego_vehicle = np.arctan2(ego_vehicle_location.y-cy,ego_vehicle_location.x-cx)
                target_vehicle_speed = get_speed(target_vehicle)
                prediction_threshold = 5
                dtheta = (-theta_target_vehicle + theta_ego_vehicle) % (2*np.pi)
                cosangle = self._direction_to_center(target_vehicle)

                if distance_to_roundabout < self._r1 + 2:
                    distance = self._r1*(dtheta)
                    if distance < front_vehicle_inside_distance:
                        front_vehicle_inside_distance = distance
                        front_vehicle_inside = target_vehicle
                        front_vehicle_inside_speed = target_vehicle_speed
                        front_vehicle_inside_direction = cosangle

                    behind_distance = self._r1*(2*np.pi-dtheta)
                    if behind_distance < behind_vehicle_inside_distance:
                        behind_vehicle_inside_distance = behind_distance
                        behind_vehicle_inside = target_vehicle
                        behind_vehicle_inside_speed = target_vehicle_speed

                    if cosangle < -0.1 and target_vehicle_speed > prediction_threshold:
                        distance = self._r2*(dtheta)
                        if distance < front_vehicle_outside_distance:
                            front_vehicle_outside_distance = distance
                            front_vehicle_outside = target_vehicle
                            front_vehicle_outside_speed = target_vehicle_speed
                            front_vehicle_outside_direction = cosangle


                if distance_to_roundabout >= self._r1 + 1 and distance_to_roundabout < self._r2 + 4:
                    distance = self._r2*(dtheta)
                    if distance < front_vehicle_outside_distance:
                        front_vehicle_outside_distance = distance
                        front_vehicle_outside = target_vehicle
                        front_vehicle_outside_speed = target_vehicle_speed
                        front_vehicle_outside_direction = cosangle

                    behind_distance = self._r2*(2*np.pi-dtheta)
                    if behind_distance < behind_vehicle_outside_distance:
                        behind_vehicle_outside_distance = behind_distance
                        behind_vehicle_outside = target_vehicle
                        behind_vehicle_outside_speed = target_vehicle_speed

                if distance_to_roundabout >= self._r1 + 1 and distance_to_roundabout < self._r2 + 13:
                    if cosangle > 0.2 and target_vehicle_speed > prediction_threshold:
                        distance = self._r1*(dtheta)
                        if distance < front_vehicle_inside_distance:
                            front_vehicle_inside_distance = distance
                            front_vehicle_inside = target_vehicle
                            front_vehicle_inside_speed = target_vehicle_speed
                            front_vehicle_inside_direction = cosangle


                if distance_to_roundabout >= self._r2 + 1 and distance_to_roundabout < self._r2 + 13:
                    if cosangle > 0.2 and target_vehicle_speed > prediction_threshold:
                        distance = self._r2*(dtheta)
                        if distance < front_vehicle_outside_distance:
                            front_vehicle_outside_distance = distance
                            front_vehicle_outside = target_vehicle
                            front_vehicle_outside_speed = target_vehicle_speed
                            front_vehicle_outside_direction = cosangle


        if front_vehicle_inside_distance > 300 and front_vehicle_outside_distance >300 and behind_vehicle_outside_distance > 300 and behind_vehicle_inside_distance > 300:
            self._empty_road = True
        else:
            self._empty_road = False

        """
        Update State
        """
        self._RL_state.ego_x = ego_vehicle_location.x
        self._RL_state.ego_y = ego_vehicle_location.y
        self._RL_state.ego_speed = get_speed(self._vehicle)
        self._RL_state.ego_vehicle = self._vehicle

        self._RL_state.front_vehicle_inside = front_vehicle_inside
        self._RL_state.front_vehicle_inside_distance = front_vehicle_inside_distance
        self._RL_state.front_vehicle_inside_direction = front_vehicle_inside_direction
        if front_vehicle_inside_distance < 200:
            self._RL_state.front_vehicle_inside_speed = front_vehicle_inside_speed
        else:
            self._RL_state.front_vehicle_inside_speed = 30.0

        self._RL_state.front_vehicle_outside = front_vehicle_outside
        self._RL_state.front_vehicle_outside_distance = front_vehicle_outside_distance
        self._RL_state.front_vehicle_outside_direction = front_vehicle_outside_direction
        if front_vehicle_outside_distance < 200:
            self._RL_state.front_vehicle_outside_speed = front_vehicle_outside_speed
        else:
            self._RL_state.front_vehicle_outside_speed = 30.0

        self._RL_state.behind_vehicle_inside = behind_vehicle_inside
        self._RL_state.behind_vehicle_inside_distance = behind_vehicle_inside_distance
        self._RL_state.behind_vehicle_inside_speed = behind_vehicle_inside_speed
    
        self._RL_state.behind_vehicle_outside = behind_vehicle_outside
        self._RL_state.behind_vehicle_outside_distance = behind_vehicle_outside_distance
        self._RL_state.behind_vehicle_outside_speed = behind_vehicle_outside_speed

        self._RL_state.front_vehicle_inside_percp = front_vehicle_inside_percp
        self._RL_state.front_vehicle_outside_percp = front_vehicle_outside_percp
        self._RL_state.behind_vehicle_inside_percp = behind_vehicle_inside_percp
        self._RL_state.behind_vehicle_outside_percp = behind_vehicle_outside_percp
        

    def _generate_decision(self):

        
        self._environment_perception()

        front_vehicle_inside = self._RL_state.front_vehicle_inside
        # front_vehicle_inside_distance = self._RL_state.front_vehicle_inside_distance
        # front_vehicle_inside_speed = self._RL_state.front_vehicle_inside_speed
        front_vehicle_outside = self._RL_state.front_vehicle_outside
        # front_vehicle_outside_distance = self._RL_state.front_vehicle_outside_distance
        # front_vehicle_outside_speed = self._RL_state.front_vehicle_outside_speed

        behind_vehicle_inside = self._RL_state.behind_vehicle_inside
        # behind_vehicle_inside_distance = self._RL_state.behind_vehicle_inside_distance
        behind_vehicle_outside = self._RL_state.behind_vehicle_outside
        # behind_vehicle_outside_distance = self._RL_state.behind_vehicle_outside_distance


        """
        Rule-based decision:
        """
        self._decision_time_accumulation += self._dt_real
        self._total_running_time += self._dt_real

        #IDM_v_inside = self._IDM_desired_speed(front_vehicle_inside,front_vehicle_inside_distance,self._dt_real)
        #IDM_v_outside = self._IDM_desired_speed(front_vehicle_outside,front_vehicle_outside_distance,self._dt_real)

        IDM_v = 15
        self._update_decision_state()

        if self._make_decision:

            r_target = self._Lane_change_decision(front_vehicle_inside,
                                front_vehicle_outside,
                                behind_vehicle_inside,
                                behind_vehicle_outside)
            self._r_target = r_target

            acc = self._longitudinal_decision(front_vehicle_inside,front_vehicle_outside)

            self._acceleration = acc
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if self._decision_time_accumulation > self._decision_dt:
            self._make_decision = True


        IDM_v = self._last_speed + self._acceleration*self._decision_dt

        if IDM_v < 0:
            IDM_v = 0
        self.set_speed(IDM_v)
        # print("%03.2f %03.2f %03.2f %03.2f %03.2f %03.2f %d" % (self._inroundabout_lane, self._r_target,IDM_v,
        #     self._decision_time_accumulation,front_vehicle_inside_distance,front_vehicle_outside_distance,self._vehicle.id))



    def record_data(self, control):

        front_inside_folder = 'data_log/front_in'
        front_outside_folder = 'data_log/front_out'
        behind_inside_folder = 'data_log/behind_in'
        behind_outside_folder = 'data_log/behind_out'
        obs_folder = 'obs'
        true_folder = 'true'
        fname_xy = 'xy_vxvy.txt'
        fname_direc_front = 'direction_frontvehonly.txt'
        fname_time = 'data_log/timestamp.txt'
        fname_control = 'data_log/control.txt'
        
        folder_list = [front_inside_folder, front_outside_folder, behind_inside_folder, behind_outside_folder]
        veh_percp_list = [self._RL_state.front_vehicle_inside_percp, 
                        self._RL_state.front_vehicle_outside_percp, 
                        self._RL_state.behind_vehicle_inside_percp, 
                        self._RL_state.behind_vehicle_outside_percp]
        speed_list = [self._RL_state.front_vehicle_inside_speed, 
                        self._RL_state.front_vehicle_outside_speed, 
                        self._RL_state.behind_vehicle_inside_speed,
                        self._RL_state.behind_vehicle_outside_speed]
        dist_list = [self._RL_state.front_vehicle_inside_distance, 
                        self._RL_state.front_vehicle_outside_distance, 
                        self._RL_state.behind_vehicle_inside_distance,
                        self._RL_state.behind_vehicle_outside_distance]

        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

        ##### observation value
        if self._perception_enable:
            for i, folder in enumerate(folder_list):
                if not os.path.exists(os.path.join(folder, obs_folder)):
                    os.makedirs(os.path.join(folder, obs_folder))
                with open(os.path.join(folder, obs_folder, fname_xy), 'a') as file:
                    if veh_percp_list[i] is None:
                        file.write('nan nan nan nan nan nan\n')
                    else:
                        x = veh_percp_list[i].location.x
                        y = veh_percp_list[i].location.y
                        vx = veh_percp_list[i].state[2]
                        vy = veh_percp_list[i].state[3]
                        speed = speed_list[i]
                        distance = dist_list[i]
                        file.write('%.2f %.2f %.2f %.2f %.2f %.2f\n'%(x, y, vx, vy, speed, distance) )

            with open(os.path.join(front_inside_folder, obs_folder, fname_direc_front), 'a') as file:
                direction = self._RL_state.front_vehicle_inside_direction
                if direction > 199:
                    file.write('nan\n')
                else:
                    file.write('%.2f \n'%(direction))
            
            with open(os.path.join(front_outside_folder, obs_folder, fname_direc_front), 'a') as file:
                direction = self._RL_state.front_vehicle_outside_direction
                if direction > 199:
                    file.write('nan\n')
                else:
                    file.write('%.2f \n'%(direction))

        ##### true value
        veh_percp_list_true = [self._RL_state.front_vehicle_inside, 
                        self._RL_state.front_vehicle_outside, 
                        self._RL_state.behind_vehicle_inside, 
                        self._RL_state.behind_vehicle_outside]

        dist_func_true = [lambda x: self._r1*(x), 
                        lambda x: self._r2*(x), 
                        lambda x: self._r1*(2*np.pi-x), 
                        lambda x: self._r2*(2*np.pi-x) ]

        for i, folder in enumerate(folder_list):
            if not os.path.exists(os.path.join(folder, true_folder)):
                os.makedirs(os.path.join(folder, true_folder))
            with open(os.path.join(folder, true_folder, fname_xy), 'a') as file:
                if veh_percp_list[i] is None:
                    file.write('nan nan nan nan nan nan\n')
                else:
                    loc = veh_percp_list_true[i].get_location()
                    velo = veh_percp_list_true[i].get_velocity()
                    speed = get_speed(veh_percp_list_true[i])

                    theta_target_vehicle = np.arctan2(loc.y-self._cy,loc.x-self._cx)
                    ego_loc = self._vehicle.get_location()
                    theta_ego_vehicle = np.arctan2(ego_loc.y-self._cy,ego_loc.x-self._cx)
                    dtheta = (-theta_target_vehicle + theta_ego_vehicle) % (2*np.pi)

                    distance = dist_func_true[i](dtheta)
                    file.write('%.2f %.2f %.2f %.2f %.2f %.2f\n'%(loc.x, loc.y, velo.x, velo.y, speed, distance) )

        with open(os.path.join(front_inside_folder, true_folder, fname_direc_front), 'a') as file:
            if self._RL_state.front_vehicle_inside is None:
                file.write('nan\n')
            else:
                direction = self._direction_to_center(self._RL_state.front_vehicle_inside)
                file.write('%.2f \n'%(direction))
        
        with open(os.path.join(front_outside_folder, true_folder, fname_direc_front), 'a') as file:
            if self._RL_state.front_vehicle_outside is None:
                file.write('nan\n')
            else:
                direction = self._direction_to_center(self._RL_state.front_vehicle_outside)
                file.write('%.2f \n'%(direction))
            
        with open(fname_time, 'a') as file:
            file.write('%.2f \n'%(self._total_running_time))

        with open(fname_control, 'a') as file:
            file.write('%.2f %.2f %.2f %.2f %.2f %.2f\n'%(self._RL_state.ego_x, self._RL_state.ego_y, self._RL_state.ego_speed, control.throttle, control.brake, control.steer))

    def update_data(self, control):

        veh_percp_list_true = [
            self._RL_state.front_vehicle_inside, 
            self._RL_state.front_vehicle_outside, 
            self._RL_state.behind_vehicle_inside, 
            self._RL_state.behind_vehicle_outside]

        dist_func_true = [lambda x: self._r1*(x), 
            lambda x: self._r2*(x), 
            lambda x: self._r1*(2*np.pi-x), 
            lambda x: self._r2*(2*np.pi-x) ]

        veh_percp_list_true_data = []

        for i, veh in enumerate(veh_percp_list_true):
            if veh is None:
                veh_percp_list_true_data.append((np.nan, np.nan))
            else:
                loc = veh.get_location()
                velo = veh.get_velocity()
                speed = get_speed(veh)

                theta_target_vehicle = np.arctan2(loc.y-self._cy,loc.x-self._cx)
                ego_loc = self._vehicle.get_location()
                theta_ego_vehicle = np.arctan2(ego_loc.y-self._cy,ego_loc.x-self._cx)
                dtheta = (-theta_target_vehicle + theta_ego_vehicle) % (2*np.pi)

                distance = dist_func_true[i](dtheta)
            
                veh_percp_list_true_data.append((speed, distance))
        
        data_append(
            time=self._total_running_time,
            throttle=control.throttle,
            brake=control.brake,
            steering=control.steer,
            speed=self._RL_state.ego_speed,

            enable_safeguard=self._rulebased_signal,
            enable_perception=self._perception_enable,
            enable_aggressive=self._aggressive,

            dist_inner_behind_true=veh_percp_list_true_data[2][1] if veh_percp_list_true_data[2][1] < 200 else np.nan,
            dist_inner_behind_perc=self._RL_state.behind_vehicle_inside_distance if self._RL_state.behind_vehicle_inside_distance < 200 else np.nan,
            dist_inner_front_true=veh_percp_list_true_data[0][1] if veh_percp_list_true_data[0][1] < 200 else np.nan,
            dist_inner_front_perc=self._RL_state.front_vehicle_inside_distance if self._RL_state.front_vehicle_inside_distance < 200 else np.nan,
            dist_outer_front_true=veh_percp_list_true_data[1][1] if veh_percp_list_true_data[1][1] < 200 else np.nan,
            dist_outer_front_perc=self._RL_state.front_vehicle_outside_distance if self._RL_state.front_vehicle_outside_distance < 200 else np.nan,
            dist_outer_behind_true=veh_percp_list_true_data[3][1] if veh_percp_list_true_data[3][1] < 200 else np.nan,
            dist_outer_behind_perc=self._RL_state.behind_vehicle_outside_distance if self._RL_state.behind_vehicle_outside_distance < 200 else np.nan,

            speed_inner_behind_true=veh_percp_list_true_data[2][0] if veh_percp_list_true_data[2][1] < 200 else np.nan,
            speed_inner_behind_perc=self._RL_state.behind_vehicle_inside_speed if self._RL_state.behind_vehicle_inside_distance < 200 else np.nan,
            speed_inner_front_true=veh_percp_list_true_data[0][0] if veh_percp_list_true_data[0][1] < 200 else np.nan,
            speed_inner_front_perc=self._RL_state.front_vehicle_inside_speed if self._RL_state.front_vehicle_inside_distance < 200 else np.nan,
            speed_outer_front_true=veh_percp_list_true_data[1][0] if veh_percp_list_true_data[1][1] < 200 else np.nan,
            speed_outer_front_perc=self._RL_state.front_vehicle_outside_speed if self._RL_state.front_vehicle_outside_distance < 200 else np.nan,
            speed_outer_behind_true=veh_percp_list_true_data[3][0] if veh_percp_list_true_data[3][1] < 200 else np.nan,
            speed_outer_behind_perc=self._RL_state.behind_vehicle_outside_speed if self._RL_state.behind_vehicle_outside_distance < 200 else np.nan,
        )


    def _generate_waypoint_location(self, waypoint, r_t):
        r1 = self._r1
        r2 = self._r2

        r = r1
        cx = -0.4771
        cy = 0.1983
        AV_x = self._vehicle.get_location().x
        AV_y = self._vehicle.get_location().y
        if abs((AV_x*AV_x+AV_y*AV_y-r2*r2))<50:
            self._inroundabout_inside = True

        if not self._inroundabout_inside:
            target_x = waypoint.transform.location.x
            target_y = waypoint.transform.location.y
            return carla.Location(target_x,target_y)

        r_target = r_t
        current_speed = get_speed(self._vehicle)

        dt = 0.2
        if self._inroundabout == 1.0 and self._r_target == self._r1:
            dt = 0.4

        if self._inroundabout == 2.0 and self._r_target == self._r2:
            dt = 0.4

        if current_speed > 36 and dt == 0.4:
            dt = 0.25

        r_vehicle = (AV_x-cx)*(AV_x-cx)+(AV_y-cy)*(AV_y-cy)
        r_vehicle = np.sqrt(r_vehicle)


        if r_target>r_vehicle:
            r_next = min(r_vehicle+(r2-r1)/self._lane_change_time*dt,r_target)
        else:
            r_next = max(r_vehicle-(r2-r1)/self._lane_change_time*dt,r_target)

        ####
        if current_speed > 40:
            r_next = r_next-1
        ####

        theta_last = np.arctan2((AV_y-cy),(AV_x-cx))
        ddd = current_speed*dt
        if ddd < 2:
            ddd = 2
        d_theta = ddd/r_vehicle
        theta_next = theta_last - d_theta
        next_x = cx + r_next*np.cos(theta_next)
        next_y = cy + r_next*np.sin(theta_next)
        target_x = next_x
        target_y = next_y

        return carla.Location(target_x,target_y)


    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # self._dt_real = self._clock.dt()

        # not enough waypoints in the horizon? => add more!
        #print(len(self._waypoints_queue))
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            if not self._global_plan:
                self._compute_next_waypoints(k=10)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    a,b = self._waypoints_queue.popleft()
                    #print(a)
                    self._waypoint_buffer.append(
                        (a,b))
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        waypoint = self._target_waypoint
    
        self._dt_real = self._clock.dt()
        self._generate_decision()

        waypoint_location = self._generate_waypoint_location(waypoint,self._r_target)

        #control = self._vehicle_controller.run_step(self._target_speed, self._target_waypoint)
        #self._target_speed = 20 #################################
        #print(self._target_speed)

        control = self._vehicle_controller.run_step(self._target_speed, waypoint_location, self._dt_real)

        # record perception and control
        self.record_data(control)
        self.update_data(control)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance +1 :
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            #return
            #draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], self._vehicle.get_location().z + 1.0)
            arrow_color=carla.Color()

            # control.throttle = 0.0
            # control.brake = 0.0
            if self._rulebased_signal:
                arrow_color.r = 25
                arrow_color.g = 202
                arrow_color.b = 173
                sz = 2
            else:
                arrow_color.r = 220
                arrow_color.g = 20
                arrow_color.b = 60
                sz = 2

            if self._inroundabout_inside:
                world_draw = self._vehicle.get_world()
                end = waypoint_location
                end.z = 0.0
                cx = -0.4771
                cy = 0.1983
                theta_end = np.arctan2((end.y-cy),(end.x-cx))
                r_end = np.sqrt((end.x-cx)*(end.x-cx)+(end.y-cy)*(end.y-cy))
                d_theta = np.pi/180*3
                theta_begin = theta_end+ d_theta
                begin = carla.Location(x = cx+r_end *np.cos(theta_begin),y = cy+r_end *np.sin(theta_begin))
                begin.z = 0.0
                # arrow_color=carla.Color()
                world_draw.debug.draw_arrow(begin, end, thickness=0.2, arrow_size = sz, color = arrow_color, life_time=1)

        return control


    """
    The following function are written for RL
    """

#########

    def get_RL_state(self):

        self._environment_perception()
        return self._RL_state


    def set_target_RL(self, action_option):
        """
        Set decision by RL
        """
        """
        action space:
        0: rule-based policy
        1: emergency brake (acc = -4)
        2: acc = 0; target to outside
        3: acc = 0; target to inside
        4: acc = 2; target to outside
        5: acc = 2; target to inside
        6: acc = -2; target to outside
        7: acc = -2; target to inside
        """
        if action_option == 1:
            #print("Emergency Braking")
            ##self._r_target = self._r2
            self._acceleration = -4
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if action_option == 2:
            self._r_target = self._r2
            self._acceleration = 0
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if action_option == 3:
            self._r_target = self._r1
            self._acceleration = 0
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if action_option == 4:
            self._r_target = self._r2
            self._acceleration = 2
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if action_option == 5:
            self._r_target = self._r1
            self._acceleration = 2
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if action_option == 6:
            self._r_target = self._r2
            self._acceleration = -2
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

        if action_option == 7:
            self._r_target = self._r1
            self._acceleration = -2
            self._last_speed = get_speed(self._vehicle)
            self._make_decision = False
            self._decision_step += 1
            self._decision_time_accumulation = 0

########
def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
