import sys, os, glob, threading, shutil, cv2

sys.path.append("../PythonAPI")
sys.path.append("../PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg")

import carla
from carla import ColorConverter as cc
# from agents.navigation.roaming_agent import *
# from agents.navigation.basic_agent import *

import numpy as np
import time

# from detect_func import *

# host = "192.168.1.13" # Local
# host = "35.3.70.70"
host = "127.0.0.1"
# host = "35.3.17.188" # zhong's net
# host = "35.3.57.207" # mcity's net
port = 2000
vehicle_id = 65

class Camera(threading.Thread):
    def __init__(self, world, image_size_x=640, image_size_y=480,
        fov=90, sensor_tick=0.0):
        # initialize yolov3
        # self.yolo = detect()
        self.frame_id = 0
        
        if os.path.exists("yolov3/carla_in"):
            shutil.rmtree("yolov3/carla_in")  # delete output folder
        os.makedirs("yolov3/carla_in")  # make new output folder
        print("entering...")

        # Add sensor
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(image_size_x))
        blueprint.set_attribute('image_size_y', str(image_size_y))
        blueprint.set_attribute('fov', str(fov))
        blueprint.set_attribute('sensor_tick', str(sensor_tick))

        vehicles = [vehicle for vehicle in world.get_actors().filter("*vehicle*") if vehicle.id == vehicle_id]
        if len(vehicles) != 1:
            raise RuntimeError("Cannot find exact vehicle")
        self._vehicle = vehicles[0]
        self._sensor_transforms = carla.Transform(carla.Location(z=2.5))
        self._sensor = world.spawn_actor(blueprint, self._sensor_transforms, attach_to=self._vehicle)
        self._sensor.listen(self.receive)

        self._visualizer = None
        self._processing = False
        threading.Thread.__init__(self)

    def __del__(self):
        self._sensor.destroy()
        # del self.yolo

    def run(self):
        while True:
            pass
        # self._visualizer = pcl.Visualizer()
        # self._visualizer.addPointCloud(pcl.PointCloud(np.random.rand(10, 3).astype('f4')), 'lidar')
        # while not self._visualizer.wasStopped():
        #     try: self._visualizer.spinOnce(50)
        #     except KeyboardInterrupt: pass

    def receive(self, message):
        # message is carla.Image object
        if self._processing: 
            # print("Skipping frame")
            return
        else:
            # print("Received frame")
            self._processing = True

        message.convert(cc.Raw)
        array = np.frombuffer(message.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (message.height, message.width, 4))
        array = np.ascontiguousarray(array[:, :, :3])
        # option 1: save to disk
        cv2.imwrite("yolov3/carla_in/frame.jpg", array)
        # option2: send to another thread
        # array = array[:, :, ::-1]
        # detections = self.yolo.run(array, frame_id = self.frame_id)

        self.frame_id += 1

        # message.save_to_disk('_out/%08d' % message.frame_number)
        
        self._processing = False


def main():
    client = carla.Client(host, port)
    client.set_timeout(4.0)
    world = client.get_world()
    Camera(world).start()

if __name__ == "__main__":
    main()

