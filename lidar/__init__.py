from subprocess import Popen, PIPE
import numpy as np
from time import time
from sklearn.linear_model import RANSACRegressor

LIDAR_EXE = "lidar/CarlaLidar"

def run_cpp(pointcloud):
    '''
    pointcloud: sensor.PointCloud type
    
    Return
    --------
    (lpoints, rpoints): points of both edges, y reverted
    (k, b): x = ky + b
    (lheadx, rheadx): x offset of both edges at head
    (ltailx, rtailx): x offset of both edges at tail
    '''
    try:
        tstart = time()
        proc = Popen([LIDAR_EXE, "-"], stdin=PIPE, stdout=PIPE) #, stderr=PIPE)

        # Feed data
        proc.stdin.write((str(len(pointcloud)) + ' ').encode('ascii'))
        for point in pointcloud.array:
            proc.stdin.write((' '.join(str(d) for d in point) + '\n').encode('ascii'))
        proc.stdin.close()

        # Read output
        if proc.wait() != 0: return False
        leftnum, rightnum = proc.stdout.readline().decode('ascii').split(' ')
        leftnum, rightnum = int(leftnum), int(rightnum)
        lpoints, rpoints = [], []
        for idx, line in enumerate(proc.stdout.readlines()):
            pt = [float(f) for f in line.decode('ascii').split(' ')]
            if idx < leftnum:
                lpoints.append(pt)
            else:
                rpoints.append(pt)
        lpoints, rpoints = np.array(lpoints), np.array(rpoints)

        # Post process
        if len(lpoints) < 3 or len(rpoints) < 3: return False
        lpoints[:, 0:2] *= -1 # invert y, z axis
        rpoints[:, 0:2] *= -1
        
        lcenter = lpoints[np.argmin(np.abs(lpoints[:, 1])), 0] # center offsets
        rcenter = rpoints[np.argmin(np.abs(rpoints[:, 1])), 0]
        lheadx = lpoints[np.argmax(lpoints[:, 1]), 0] - lcenter # x offsets
        rheadx = rpoints[np.argmax(rpoints[:, 1]), 0] - rcenter
        ltailx = lpoints[np.argmin(lpoints[:, 1]), 0] - lcenter
        rtailx = rpoints[np.argmin(rpoints[:, 1]), 0] - rcenter

        ransac = RANSACRegressor() # fit line
        ransac.fit(lpoints[:, [1]], lpoints[:, [0]])
        bl, kl = ransac.predict([[0], [1]])
        kl -= bl
        ransac.fit(rpoints[:, [1]], rpoints[:, [0]])
        br, kr = ransac.predict([[0], [1]])
        kr -= br
        
        k = (kl * len(lpoints) + kr * len(rpoints)) / (len(lpoints) + len(rpoints))
        b = 0.25 * bl + 0.75 * br

        tend = time()
        # print("Process time: ", tend - tstart)
        return (lpoints, rpoints), (k, b), (lheadx, rheadx), (ltailx, rtailx)
    except:
        return False
