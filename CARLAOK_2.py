#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function
from lidar import run_cpp

import argparse
import logging
import random
import time
import math

import matplotlib.pyplot as plt
import numpy as np

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


def run_carla_client(args):
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
                NumberOfVehicles=20,                         # 20
                NumberOfPedestrians=40,                      # 40
                WeatherId=1,                              # random.choice([1, 3, 7, 8, 14]),
                QualityLevel=args.quality_level)
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

            if args.lidar:
                lidar = Lidar('Lidar32')
                lidar.set_position(0, 0, 2.50)
                lidar.set_rotation(0, 0, 0)
                lidar.set(
                    Channels=32,
                    Range=50,
                    PointsPerSecond=100000,
                    RotationFrequency=10,
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
        player_start = 1

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode...')
        client.start_episode(player_start)
        px_l = -1
        py_l = -1
        # Iterate every frame in the episode.
        for frame in range(0, frames_per_episode):

            # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()

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

                if speed > 28:
                    br = (speed-28)*0.1
                    thr = 0
                else:
                    br = 0
                    thr = (28-speed)*0.05 + 0.6





                L = sensor_data["Lidar32"].point_cloud.array
                L = L[L[:,2]<2.1]
                L = L[L[:,1]<0]
                if len(L) == 0:
                    come_back = True
                else:
                    come_back = False

                if not come_back:
                    X = L[:,0]
                    Y = -L[:,1]
                    theta = math.atan2(-delta_y,delta_x)
                    delta_x = speed*math.cos(theta)*0.1/3.6
                    XX = math.cos(theta)*X - math.sin(theta)*Y
                    YY = math.sin(theta)*X + math.cos(theta)*Y
                    plt.plot(XX, YY, 'ro')
                    if min(YY)>delta_x*30:
                        come_back = True
                else:
                    theta = 0
                #plt.plot([106-pos_y,106-pos_y],[-90-pos_x,100-pos_x])
                #plt.plot([109.5-pos_y,109.5-pos_y],[-90-pos_x,100-pos_x])

                plt.plot([104-pos_y,104-pos_y],[-90-pos_x,200-pos_x])
                plt.plot([108-pos_y,108-pos_y],[-90-pos_x,200-pos_x])
                plt.plot([112-pos_y,112-pos_y],[-90-pos_x,200-pos_x])
                plt.plot([0],[0], '*')
                plt.axis([-5, 5, -10, 50])

                if come_back:
                    tx = 109.5-pos_y
                    ty = delta_x*15
                else:
                    if pos_x<0:
                        tx = min(XX)-3
                        ty = delta_x*30
                    if pos_x>=0 and pos_x<15:
                        tx = 109.5-pos_y
                        ty = delta_x*30
                    if pos_x>=15 and pos_x<32:
                        tx = 107-pos_y
                        ty = delta_x*30
                    tx = min(XX)-3
                    ty = delta_x*30
                    if ty > min(YY) and tx > 1 and np.sqrt(tx*tx+ty*ty)>5:
                        ty = min(YY)-1

                if speed < 2:
                    st = 0
                else:
                    st = -(math.atan2(-tx,ty)-math.atan2(-delta_y,delta_x))*0.8

                plt.plot([tx],[ty], 'r*')

                if pos_x>60:
                    br = 1
                    thr = 0
                    st = 0



                if frame == 0:
                    plt.show(block=False)
                    plt.pause(5)
                    plt.clf()
                else:
                    plt.show(block=False)
                    plt.pause(0.01)
                    plt.clf()

                if abs(st)<0.001:
                    st = 0
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
