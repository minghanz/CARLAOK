import threading
import socket
import msgpack
import math
import numpy as np

class LidarSubscriber(threading.Thread):
    def __init__(self, port=6660):
        threading.Thread.__init__(self)
        self.data = []
        self.time = None
        self._receiver = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
        self._receiver.bind(("127.0.0.1", port))

    def run(self):
        while True:
            # print("Waiting for data...")
            raw, addr = self._receiver.recvfrom(8192) # buffer size is 8192 bytes
            self.data, self.time = msgpack.unpackb(raw)
            print("Received %d objects from Lidar" % len(self.data))

    def get_global(self, vehicle_transform):
        # self.data = [(5,-15)] # for debug
        if len(self.data) == 0:
            return np.array([]).reshape(0,3)

        ego_vehicle_location = vehicle_transform.location
        AV_x = ego_vehicle_location.x
        AV_y = ego_vehicle_location.y

        sy, cy = math.sin(math.radians(vehicle_transform.rotation.yaw)), math.cos(math.radians(vehicle_transform.rotation.yaw))
        rot = np.array([[cy, sy], [-sy, cy]])
        arr = np.array(self.data)

        # post processing
        darr = np.full((len(arr), 1), self.time)
        arr = arr[:,:2].dot(np.array([[0,1],[-1,0]])) # fix coordinate
        arr = arr.dot(rot) + np.array([[ego_vehicle_location.x, ego_vehicle_location.y]])
        return np.hstack((darr, arr))

class CameraSubscriber(threading.Thread):
    def __init__(self, port=6661):
        threading.Thread.__init__(self)
        self.data = []
        self._receiver = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
        self._receiver.bind(("127.0.0.1", port))

    def run(self):
        while True:
            # print("Waiting for data...")
            raw, addr = self._receiver.recvfrom(8192) # buffer size is 8192 bytes
            self.data = msgpack.unpackb(raw)
            print("Received %d objects from Camera" % len(self.data))

    def get_global(self, vehicle_transform):
        # self.data = [(5,-15)] # for debug
        if len(self.data) == 0:
            return np.array([]).reshape(0,3), np.array([]).reshape(0,3)

        ego_vehicle_location = vehicle_transform.location
        AV_x = ego_vehicle_location.x
        AV_y = ego_vehicle_location.y

        sy, cy = math.sin(math.radians(vehicle_transform.rotation.yaw)), math.cos(math.radians(vehicle_transform.rotation.yaw))
        rot = np.array([[cy, sy], [-sy, cy]])
        arr = np.array(self.data)

        # post processing
        darr = np.full((len(arr), 1), 0)
        arr = arr[:,:2].dot(np.array([[0,1],[1,0]])) # fix coordinate
        local_arr = np.copy(arr)
        arr = arr.dot(rot) + np.array([[ego_vehicle_location.x, ego_vehicle_location.y]])
        return np.hstack((darr, arr)), local_arr




lidar = LidarSubscriber()
lidar.start()
camera = CameraSubscriber()
camera.start()

