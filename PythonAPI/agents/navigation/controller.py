#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math

import numpy as np


import carla
from agents.tools.misc import distance_vehicle, get_speed

# class VehiclePIDController():
#     """
#     VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
#     low level control a vehicle from client side
#     """

#     def __init__(self, vehicle,
#                  args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
#                  args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}):
#         """
#         :param vehicle: actor to apply to local planner logic onto
#         :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
#                              K_P -- Proportional term
#                              K_D -- Differential term
#                              K_I -- Integral term
#         :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
#         semantics:
#                              K_P -- Proportional term
#                              K_D -- Differential term
#                              K_I -- Integral term
#         """
#         self._vehicle = vehicle
#         self._world = self._vehicle.get_world()
#         self._lon_controller = PIDLongitudinalController(
#             self._vehicle, **args_longitudinal)
#         self._lat_controller = PIDLateralController(
#             self._vehicle, **args_lateral)

#     def run_step(self, target_speed, waypoint):
#         """
#         Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
#         at a given target_speed.

#         :param target_speed: desired vehicle speed
#         :param waypoint: target location encoded as a waypoint
#         :return: distance (in meters) to the waypoint
#         """
#         throttle = self._lon_controller.run_step(target_speed)
#         steering = self._lat_controller.run_step(waypoint)

#         thr = 0
#         br = 0
#         current_speed = get_speed(self._vehicle)
#         if current_speed > 20:
#             thr = 0
#             br = 0.1*(current_speed-20)
#         else:
#             thr = 0.1*(20-current_speed)
#             br = 0

#         control = carla.VehicleControl()
#         control.steer = steering
#         control.throttle = thr #throttle 
#         control.brake = br #0.0
#         control.hand_brake = False
#         control.manual_gear_shift = False

#         return control


# class PIDLongitudinalController():
#     """
#     PIDLongitudinalController implements longitudinal control using a PID.
#     """

#     def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
#         """
#         :param vehicle: actor to apply to local planner logic onto
#         :param K_P: Proportional term
#         :param K_D: Differential term
#         :param K_I: Integral term
#         :param dt: time differential in seconds
#         """
#         self._vehicle = vehicle
#         self._K_P = K_P
#         self._K_D = K_D
#         self._K_I = K_I
#         self._dt = dt
#         self._e_buffer = deque(maxlen=30)

#     def run_step(self, target_speed, debug=False):
#         """
#         Execute one step of longitudinal control to reach a given target speed.

#         :param target_speed: target speed in Km/h
#         :return: throttle control in the range [0, 1]
#         """
#         current_speed = get_speed(self._vehicle)

#         if debug:
#             print('Current speed = {}'.format(current_speed))

#         return self._pid_control(target_speed, current_speed)

#     def _pid_control(self, target_speed, current_speed):
#         """
#         Estimate the throttle of the vehicle based on the PID equations

#         :param target_speed:  target speed in Km/h
#         :param current_speed: current speed of the vehicle in Km/h
#         :return: throttle control in the range [0, 1]
#         """
#         _e = (target_speed - current_speed)
#         self._e_buffer.append(_e)

#         if len(self._e_buffer) >= 2:
#             _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
#             _ie = sum(self._e_buffer) * self._dt
#         else:
#             _de = 0.0
#             _ie = 0.0

#         return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), 0.0, 1.0)


# class PIDLateralController():
#     """
#     PIDLateralController implements lateral control using a PID.
#     """

#     def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
#         """
#         :param vehicle: actor to apply to local planner logic onto
#         :param K_P: Proportional term
#         :param K_D: Differential term
#         :param K_I: Integral term
#         :param dt: time differential in seconds
#         """
#         self._vehicle = vehicle
#         self._K_P = K_P
#         self._K_D = K_D
#         self._K_I = K_I
#         self._dt = dt
#         self._e_buffer = deque(maxlen=10)

#     def run_step(self, waypoint):
#         """
#         Execute one step of lateral control to steer the vehicle towards a certain waypoin.

#         :param waypoint: target waypoint
#         :return: steering control in the range [-1, 1] where:
#             -1 represent maximum steering to left
#             +1 maximum steering to right
#         """
#         return self._pid_control(waypoint, self._vehicle.get_transform())

#     def _pid_control(self, waypoint, vehicle_transform):
#         """
#         Estimate the steering angle of the vehicle based on the PID equations

#         :param waypoint: target waypoint
#         :param vehicle_transform: current transform of the vehicle
#         :return: steering control in the range [-1, 1]
#         """
#         v_begin = vehicle_transform.location
#         v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
#                                          y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

#         v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])


# ################################
#         inroundabout = False
#         r = 19.8
#         cx = -0.4771
#         cy = 0.1983
#         AV_x = vehicle_transform.location.x
#         AV_y = vehicle_transform.location.y

#         if abs((AV_x*AV_x+AV_y*AV_y-r*r))<50:
#             inroundabout = True
#         current_speed = get_speed(self._vehicle)
#         dt = 0.2
        
#         theta_last = np.arctan2((AV_y-cy),(AV_x-cx))
#         d_theta = current_speed*dt/r
#         theta_next = theta_last - d_theta
#         next_x = cx + r*np.cos(theta_next)
#         next_y = cy + r*np.sin(theta_next)
        
#         target_x = waypoint.transform.location.x
#         target_y = waypoint.transform.location.y
#         if inroundabout:
#             target_x = next_x
#             target_y = next_y
# ##################################
#         w_vec = np.array([target_x-
#                           v_begin.x, target_y -
#                           v_begin.y, 0.0])
#         _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
#                          (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

#         _cross = np.cross(v_vec, w_vec)
#         if _cross[2] < 0:
#             _dot *= -1.0

#         self._e_buffer.append(_dot)
#         if len(self._e_buffer) >= 2:
#             _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
#             _ie = sum(self._e_buffer) * self._dt
#         else:
#             _de = 0.0
#             _ie = 0.0

#         return np.clip((self._K_P * _dot) + (self._K_D * _de /
#                                              self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)

#######################################


class VehiclePIDController_Nowaypoint():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle,
                 args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0},
                 args_longitudinal={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController(
            self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(
            self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint, real_dt):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        self._lon_controller._dt = real_dt
        self._lat_controller._dt = real_dt

        throttle, brake = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(waypoint)


        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle 
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=2.0, dt=0.02):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = 0.25/3.6#K_P
        self._K_D = 0.01#0.05#0.02 #K_D
        self._K_I = 0.012#5/200#0.155628 #K_I
        self._integ = 0.0
        self._dt = 0.05#dt
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

        :param target_speed: target speed in Km/h
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._integ += _e * self._dt
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = self._integ
            
            #print("%.4f,%.4f" % (_ie,current_speed))
        else:
            _de = 0.0
            _ie = 0.0
        
        kp = self._K_P
        ki = self._K_I
        kd = self._K_D

        if target_speed < 5:
            ki = 0
            kd = 0
        # if target_speed < 3:
        #     self._integ = 0

        calculate_value = np.clip((kp * _e) + (kd * _de) + (ki * _ie), -1.0, 1.0)
        if calculate_value > 0:
            thr = calculate_value
            br = 0
        else:
            thr = 0
            br = -calculate_value
            

        return thr,br


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

        target_x = waypoint.x
        target_y = waypoint.y

        w_vec = np.array([target_x-
                          v_begin.x, target_y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        ##################
        lf = 1.2
        lr = 1.65
        lwb = lf+lr
        
        v_rear_x = v_begin.x - v_vec[0]*lr/np.linalg.norm(v_vec)
        v_rear_y = v_begin.y - v_vec[1]*lr/np.linalg.norm(v_vec)
        l = (target_x-v_rear_x)*(target_x-v_rear_x)+(target_y-v_rear_y)*(target_y-v_rear_y)
        l = math.sqrt(l)

        theta = np.arctan(2*np.sin(_dot)*lwb/l)
        ##################

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        # theta2 = np.clip((self._K_P * _dot) + (self._K_D * _de /
        #                                      self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)
        #print(theta,theta2)

        k = 1# np.pi/180*50
        theta = theta*k
        return theta