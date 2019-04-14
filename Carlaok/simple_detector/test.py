import detector
from pcl.io import load_ply, load_pcd
from pcl import Visualizer

def visualize(detections):
    vis = Visualizer()
    for i, d in enumerate(detections):
        # print(len(d.cloud), d.h, d.w, d.l)
        # if d.h*d.w*d.l > 5000: continue
        vis.addPointCloud(d.cloud, id="obj%02d" % i)
        vis.addSphere([d.x,d.y,d.z], 0.2, id="cobj%02d" % i)
    vis.spin()
    vis.close()

def main(cluster_thres=0.5, ransac_thres=0.3):
    pcddir = "/home/jacobz/Carla/binary_0.9.4/_out/test.pcd"
    cloud = load_pcd(pcddir)
    detections = detector.run_detect(cloud, cluster_thres, ransac_thres)
    visualize(detections)

if __name__ == "__main__":
    main()
