#ifndef SIMPLE_DETECTOR_BINDING
#define SIMPLE_DETECTOR_BINDING

struct Detection3D
{
    float x, y, z;
    float h, w, l;
    float yaw;
};

#include <pcl/point_types.h>
#include <pcl/conversions.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;

void detect(const PointCloudType &cloud, std::vector<Detection3D> &dobjects, std::vector<PointCloudType::Ptr> &clouds,
    float cluster_thres, float ransac_thres);
void estimate(const PointCloudType::Ptr object, Detection3D &bound);

#endif // SIMPLE_DETECTOR_BINDING
