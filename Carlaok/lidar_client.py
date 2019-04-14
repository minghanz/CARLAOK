import sys, os, glob, threading

sys.path.append("../PythonAPI")
sys.path.append("../PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg")

import carla
import detector
from carla import ColorConverter as cc

import numpy as np
import pcl
import time
import queue
import socket
import msgpack

host = "127.0.0.1" # Localhost
# host = "192.168.1.13" # Local net
# host = "35.3.17.188" # zhong's net
# host = "35.3.57.207" # mcity's net
port = 2000
vehicle_id = 65

radin = 18
radout = 24
radoff = 4
import time

class WorldClock:
    def __init__(self, world):
        self.time = 0
        self.lasttime = None
        world.on_tick(self.tick)

    def tick(self, timestamp):
        self.time = timestamp.elapsed_seconds
        
class Lidar(threading.Thread):
    def __init__(self, world, channels=64, lrange=10000,
        points_per_frame=40000, fps=20, vis=True):
        # Add sensor
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('channels', str(channels))
        blueprint.set_attribute('range', str(lrange))
        blueprint.set_attribute('points_per_second', str(points_per_frame * fps))
        blueprint.set_attribute('rotation_frequency', str(fps))

        vehicles = [vehicle for vehicle in world.get_actors().filter("*vehicle*") if vehicle.id == vehicle_id]
        if len(vehicles) != 1:
            raise RuntimeError("Cannot find exact vehicle")
        self._vehicle = vehicles[0]
        self._sensor_transforms = carla.Transform(carla.Location(z=2.5))
        self._sensor = world.spawn_actor(blueprint, self._sensor_transforms, attach_to=self._vehicle)
        self._sensor.listen(self.receive)

        self._visualizer = None
        self._dovis = vis
        self._vistext_queue = queue.Queue()
        self._processing = False
        self._publisher = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        self._clock = WorldClock(world)

        threading.Thread.__init__(self)

    def run(self):
        if self._dovis:
            self._visualizer = pcl.Visualizer()
            self._visport1 = self._visualizer.createViewPort(0, 0, 0.5, 1)
            self._visport2 = self._visualizer.createViewPort(0.5, 0, 1, 1)
            self._visualizer.addCoordinateSystem(viewport=self._visport1, scale=3)
            self._visualizer.addCoordinateSystem(viewport=self._visport2, scale=3)
            self._visualizer.addPointCloud(pcl.PointCloud(np.random.rand(10, 3).astype('f4')), 'lidar', viewport=self._visport2)
            while not self._visualizer.wasStopped():
                while not self._vistext_queue.empty():
                    t,x,y,z = self._vistext_queue.get()
                    self._visualizer.addText3D(t, [x,y,z], viewport=self._visport1, id="t"+t)
                self._visualizer.spinOnce(50)
        else:
            while True:
                pass

    def destroy(self):
        if self._sensor is not None:
            self._sensor.destroy()
            print("Destroyed lidar")
        self._visualizer.close()

    def receive(self, message):
        # message.save_to_disk('_out/%08d' % message.frame_number)
        time = self._clock.time
        points = np.frombuffer(message.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))

        if self._processing: 
            print("Skipping cloud:", len(points), " at ", time)
            return
        else:
            print("Received cloud:", len(points), " at ", time)
            self._processing = True

        # Preprocess
        circle_incenter = np.array([-radin-radoff, 0])
        circle_outcenter = np.array([-radout+radoff, 0])
        inmask = np.linalg.norm(points[:,:2] - circle_incenter, axis=1) > radin
        outmask = np.linalg.norm(points[:,:2] - circle_outcenter, axis=1) < (radout + 20) # process additional 20m in radius
        points = points[inmask & outmask] # remove points alway from the circle

        # Update overlook
        cloud = pcl.PointCloud(points)
        if self._dovis:
            self._visualizer.updatePointCloud(cloud, 'lidar')
            self._visualizer.removeAllShapes(viewport=self._visport1)
            self._visualizer.removeAllPointClouds(viewport=self._visport1)

        # Run segmentation
        detections = detector.run_detect(cloud, 2)
        send_info = []
        print("Objects:", len(detections))
        for i, scloud in enumerate(detections):
            if(len(scloud) > 500): continue
            array = np.array(scloud.xyz)

            # Estimate the box and orientation
            # l,w,x,y,a = self.estimate(array)
            # print(x,y,z,h,w,l)
            # if(w>5 or l>5): continue

            # Remove objects by size
            x, y, z = np.mean(array, axis=0)
            size = np.linalg.norm(array - np.array([x,y,z]))
            if size < 3: continue

            # Remove objects above ground
            zmin = np.min(array[:,2])
            h = np.max(array[:,2]) - zmin
            z = zmin + h/2
            if h>3 or h<0.5: continue
            if z>3 or z<0: continue

            # Visualize
            if self._dovis:
                self._visualizer.addPointCloud(scloud, id="cobj%02d" % i, viewport=self._visport1)
                self._visualizer.addArrow([0,0,0], [x,y,z], id="bobj%02d" % i, viewport=self._visport1)
                self._vistext_queue.put((str(size), x, y, z))

            # Send result
            send_info.append((float(x),float(y),float(z)))
            print("Data sent")

        self._publisher.sendto(msgpack.packb((send_info, time)), ("127.0.0.1", 6660))
        self._processing = False

    def estimate(self, cloud, cluster_thres=2, ransac_thres=0.3):
        array = cloud
        angles = np.linspace(-np.pi/4, np.pi/4, 10)
        best = (np.inf,)
        for angle in angles:
            rot = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
            rotated = rot.dot(array[:,:2].T)
            rmax, rmin = np.max(rotated, axis=0), np.min(rotated, axis=0)
            l = rmax[0] - rmin[0]
            w = rmax[1] - rmin[1]
            if l*w < best[0]:
                x = rmin[0] + l/2
                y = rmin[1] + w/2
                best = (l*w, l, w, x, y, angle)

        rot = np.array([[np.cos(best[5]), -np.sin(best[5])],[np.sin(best[5]), np.cos(best[5])]])
        x, y = rot.dot(np.array([[x],[y]]))
        return best[1],best[3],x[0],y[0],best[5]

def main():
    lidar = None
    client = carla.Client(host, port)
    client.set_timeout(4.0)
    world = client.get_world()

    try:
        lidar = Lidar(world, vis=False)
        lidar.start()
        lidar.join()
    finally:
        if lidar is not None:
            lidar.destroy()

if __name__ == "__main__":
    main()

