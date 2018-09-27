#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time

import os
# from time import sleep
import struct     # for parsing binaries to float

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from subprocess import Popen

from lidar import run_cpp

import numpy as np
import matplotlib.pyplot as plt

FIFO_IMAGES = "/home/minghanz/catkin_ws/src/carla/PythonClient/CARLAOK/images.fifo"
FIFO_LANES =  "/home/minghanz/catkin_ws/src/carla/PythonClient/CARLAOK/lanes.fifo"

def send_image(pixelarray):
    '''
    The function is to send image to c++ process through named pipe
    '''
    #print("Entering sending images")
    image_string = pixelarray.tostring()
    with open(FIFO_IMAGES, "wb") as f:
        f.write(image_string)
    #print("Exiting sending images")

    #print("Entering receiving lanes")
    a = np.ndarray((6,),float)
    with open(FIFO_LANES, "rb") as g:
        for i in range(0,6):
            data = g.read(4)
            flo = struct.unpack('f', data)
            # print(flo)
            a[i] = flo[0]
    #print("Exiting receiving lanes")

    return a


def run_carla_client(args):
    lane_detection = Popen(["/home/minghanz/lane-detection/newcheck/SFMotor/lane detection/lane-detection"],
                            cwd="/home/minghanz/lane-detection/newcheck/SFMotor/lane detection")

    # Here we will run 3 episodes with 300 frames each.
    # number_of_episodes = 3
    frames_per_episode = 10000

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        # for episode in range(0, number_of_episodes):
           # Start a new episode.

        if args.settings_filepath is None:

            # Create a CarlaSettings object. This object is a wrapper around
            # the CarlaSettings.ini file. Here we set the configuration we
            # want for the new episode.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=0,                         # 20
                NumberOfPedestrians=0,                      # 40
                WeatherId=1,                              # random.choice([1, 3, 7, 8, 14]),
                QualityLevel=args.quality_level)
            settings.randomize_seeds()

            # Now we want to add a couple of cameras to the player vehicle.
            # We will collect the images produced by these cameras every
            # frame.

            # The default camera captures RGB images of the scene.
            camera0 = Camera('CameraRGB', PostProcessing='None')
            # Set image resolution in pixels.
            camera0.set_image_size(800, 600)
            # Set its position relative to the car in meters.
            camera0.set_position(0.3, 0, 2.30) # 0.3, 0, 1.3
            settings.add_sensor(camera0)

            # Let's add another camera producing ground-truth depth.
            # camera1 = Camera('CameraDepth', PostProcessing='Depth')
            # camera1.set_image_size(800, 600)
            # camera1.set_position(0.30, 0, 1.30)
            # settings.add_sensor(camera1)

            if args.lidar:
                lidar = Lidar('Lidar32')
                lidar.set_position(0, 0, 2.50)
                lidar.set_rotation(0, 0, 0)
                lidar.set(
                    Channels=32,
                    Range=50,
                    PointsPerSecond=100000,
                    RotationFrequency=20,
                    UpperFovLimit=10,
                    LowerFovLimit=-30)
                settings.add_sensor(lidar)

        else:

            # Alternatively, we can load these settings from a file.
            with open(args.settings_filepath, 'r') as fp:
                settings = fp.read()

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Choose one player start at random.
        # number_of_player_starts = len(scene.player_start_spots)
        # player_start = random.randint(0, max(0, number_of_player_starts - 1))
        player_start = 31

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode...')
        client.start_episode(player_start)
        px_l = -1
        py_l = -1
        ######### Parameters for one test
        lc_num = 0
        lc_hold = False
        lc_start_turn = False
        lc_num_2 = 100 # for counting failures in turning
        lc_num_3 = 0
        #############


        # Iterate every frame in the episode.
        for frame in range(0, frames_per_episode):

            # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()

            #ZYX LIDAR
            if True: #not lc_hold:
                result = run_cpp(sensor_data['Lidar32'].point_cloud, True)
                #print('LP:',LP,'END')
                
                lidar_success = False
                if result:
                    lidar_success = True
                    pts, kb, hdx, tlx = result

            # print('pts:',pts)
            # print('kb:',kb)
            # print('hdx:',hdx)
            # print('tlx:',tlx)
            print('------------------------')

            #ZYX LIDAR


            image = sensor_data['CameraRGB'].data
            lane_coef = send_image(image)
            print("lane_coef: ", lane_coef)

            # Print some of the measurements.
            #print_measurements(measurements)

            # Save the images to disk if requested.
            if args.save_images_to_disk:
                for name, measurement in sensor_data.items():
                    filename = args.out_filename_format.format(0, name, frame) # episode
                    measurement.save_to_disk(filename)

            # We can access the encoded data of a given image as numpy
            # array using its "data" property. For instance, to get the
            # depth value (normalized) at pixel X, Y
            #
            #     depth_array = sensor_data['CameraDepth'].data
            #     value_at_pixel = depth_array[Y, X]
            #

            # Now we have to send the instructions to control the vehicle.
            # If we are in synchronous mode the server will pause the
            # simulation until we send this control.

            if not args.autopilot:
                player_measurements = measurements.player_measurements
                pos_x=player_measurements.transform.location.x
                pos_y=player_measurements.transform.location.y
                speed=player_measurements.forward_speed * 3.6

                #Traffic Light
                # print('TrafficLight:-----------------')
                # for agent in measurements.non_player_agents:
                #     if agent.HasField('traffic_light'):
                #         print(agent.id)
                #         print(agent.traffic_light.transform)
                #         print(agent.traffic_light.state)
                #         print('-----------------')
                #         break

                #Traffic Light End



                if px_l == -1:
                    px_l = pos_x
                if py_l == -1:
                    py_l = pos_y
                delta_x = pos_x-px_l
                delta_y = pos_y-py_l
                st = 0
                #if pos_y < 11:
                 #   st = 0.3

                # if lidar_success and hdx[0]<-2.2: 
                #     lc_num += 1
                # else:
                #     lc_num = 0

                # if lc_hold:
                #     st = 0.3

                # if lc_num>2 and not lc_hold:
                #     lc_hold = True
                #     st = 0.2


                # if  abs(lane_coef[2]) > 0.003 and frame > 100:
                #     lc_num += 1
                # else:
                #     lc_num = 0

                # if lc_num>5 and not lc_start_turn:
                #     lc_start_turn = True

                # if lc_start_turn and lane_coef[0] == 0:
                #     lc_num_2 += 1
                
                # if lc_hold:
                #     st = 0.3

                # if lc_num_2 > 5 and not lc_hold:
                #     lc_hold = True
                #     st = 0.25



                if  abs(lane_coef[2]) > 0.003 and frame > 300:
                    lc_num += 1
                else:
                    lc_num = 0

                if lc_num>5 and not lc_start_turn:
                    lc_start_turn = True
                    lc_num_2 = 15
                
                if lc_start_turn and lc_num_2 > 0:
                    lc_num_2 -= 1
                
                if lc_hold:
                    st = 0.3

                if lc_num_2 == 0 and not lc_hold:
                    lc_hold = True
                    st = 0.25
                

                # print('lidar_success:', lidar_success )
                print('lc_num:',lc_num)
                print('lc_num_2:',lc_num_2)

                if lc_hold and lane_coef[0] != 0:
                    a1 = lane_coef[1]+lane_coef[4]
                    a2 = lane_coef[0]+lane_coef[3] - 0.2
                    l = 5
                    k = 0.08
                    st = k*(a1*l+a2)
                    print('a1:',a1)
                    print('a2:',a2)
                   
                
                

                if speed > 28:
                    br = (speed-28)*0.1
                    thr = 0
                else:
                    br = 0
                    thr = (28-speed)*0.05 + 0.6
                if pos_y > 150:
                    thr = 1.6
                    br = 0

                if lc_hold:
                    lc_num_3 += 1
                print('lc_num_3:',lc_num_3)

                if lc_num_3 > 185:
                    thr = 0
                    br = 1
                    st = 0
               # if pos_x > 5:
               #     st = -delta_y*10 + (2-pos_y)*0.8
               # if abs(st)<0.001:
              #      st = 0
                    #st = (2-pos_y)*0.01
                print('Steering:',st)
                client.send_control(
                    #steer=random.uniform(-1.0, 1.0),
                    steer=st,
                    throttle=thr,
                    brake=br,
                    hand_brake=False,
                    reverse=False)
                px_l = pos_x
                py_l = pos_y
            else:

                # Together with the measurements, the server has sent the
                # control that the in-game autopilot would do this frame. We
                # can enable autopilot by sending back this control to the
                # server. We can modify it if wanted, here for instance we
                # will add some noise to the steer.

                control = measurements.player_measurements.autopilot_control
                control.steer += random.uniform(-0.1, 0.1)
                client.send_control(control)
    
    lane_detection.kill()


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
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
        action='store_false',
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

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    try:
        os.mkfifo(FIFO_IMAGES)
    except OSError:
        pass
    try:
        os.mkfifo(FIFO_LANES)
    except OSError:
        pass

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
