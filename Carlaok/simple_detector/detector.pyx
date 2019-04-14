from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from pcl.PointCloud cimport PointCloud
from pcl cimport PointCloud as cPointCloud, PointXYZ, PCLPointCloud2
from pcl.common.conversions cimport fromPCLPointCloud2, toPCLPointCloud2
from pcl._boost cimport shared_ptr

ctypedef PointXYZ PointType
ctypedef cPointCloud[PointType] PointCloudType

cdef extern from "detector_src.h":
    cdef cppclass Detection3D:
        float x
        float y
        float z
        float h
        float w
        float l
    void detect(const PointCloudType &cloud, vector[Detection3D] &dobjects, vector[shared_ptr[PointCloudType]] &clouds, float, float)

# class Detection:
#     def __init__(self, cloud):
#         self.x = 0
#         self.y = 0
#         self.z = 0
#         self.h = 0
#         self.w = 0
#         self.l = 0
#         self.cloud = cloud

def run_detect(PointCloud cloud, cluster_thres=0.5, ransac_thres=0.3):
    cdef vector[Detection3D] result
    cdef vector[shared_ptr[PointCloudType]] clouds
    cdef PointCloudType ccloud
    fromPCLPointCloud2(deref(cloud.ptr()), ccloud)
    detect(ccloud, result, clouds, cluster_thres, ransac_thres)
    detections = []
    for i in range(clouds.size()):
        dc = PointCloud()
        dc._ptype = b'XYZ'
        toPCLPointCloud2(deref(clouds[i]), deref(dc.ptr()))
        # d = Detection(dc)
        # d.x = result[i].x
        # d.y = result[i].y
        # d.z = result[i].z
        # d.h = result[i].h
        # d.w = result[i].w
        # d.l = result[i].l
        detections.append(dc)
    return detections

