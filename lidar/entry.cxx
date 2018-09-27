#include <iostream>
#include <string>
#include <map>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/concave_hull.h>
#include <ctime>

#define PROFILE

using namespace std;
using namespace pcl;

int main(int argc, char *argv[])
{
#ifdef PROFILE
    clock_t t_start = clock();
#endif

    int retcode = 0;
    if (argc != 2)
    {
        cerr << "Unknown commandline parameters!" << endl;
        retcode = 1;
        return retcode;
    }

    const string path(argv[1]);
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    bool embed = path == "-";
    if (!embed)
        io::loadPLYFile(path, *cloud);
    else
    {
        // Read from stdin Pipe
        int number;
        cin >> number;
        for (int i = 0; i < number; i++)
        {
            PointXYZ point;
            cin >> point.x >> point.y >> point.z;
            cloud->push_back(point);
        }
    }

    // Get road plane
    const float step = 0.05;
    const int road_min_count = 500;
    map<int, vector<int>> planes;
    for (int idx = 0; idx < cloud->size(); idx++)
    {
        int hkey = int(cloud->at(idx).z / step);
        if (planes.find(hkey) == planes.end())
            planes.emplace(hkey, vector<int>());
        planes[hkey].push_back(idx);
    }

    PointCloud<PointXYZ>::Ptr road(new PointCloud<PointXYZ>);
    for (auto piter = planes.rbegin(); piter != planes.rend(); piter++)
    {
        if (piter->second.size() > road_min_count)
        {
            for (auto iter = piter->second.begin(); iter != piter->second.end(); iter++)
                road->push_back(cloud->at(*iter));
            break;
        }
    }
    cerr << "Fetched road plane";
#ifdef PROFILE
    cerr << "\tCheckpoint: " << (float)(clock() - t_start) / CLOCKS_PER_SEC << endl;
    t_start = clock();
#endif

    // Clean outliers
    PointCloud<PointXYZ>::Ptr cleanroad(new PointCloud<PointXYZ>);
    StatisticalOutlierRemoval<PointXYZ> filter;
    filter.setInputCloud(road);
    filter.setMeanK(10);
    filter.setStddevMulThresh(0.4);
    filter.filter(*cleanroad);
    cerr << " -> Cleaned outliers";
#ifdef PROFILE
    cerr << "\tCheckpoint: " << (float)(clock() - t_start) / CLOCKS_PER_SEC << endl;
    t_start = clock();
#endif

    /*********************************************
    PointXYZ minpt, maxpt;
    getMinMax3D(*cloud, minpt, maxpt);

    // Find circles
    float tfov = 10, bfov = -30;
    float dfov = (bfov - tfov) / 180 * M_PI_2 / 32;
    unordered_map<int, vector<PointXYZ>> circles;
    for (auto iter = cleanroad->begin(); iter != cleanroad->end(); iter++)
    {
        float phi = atan2(sqrt(iter->x * iter->x + iter->y * iter->y), iter->z);
        int iphi = int(phi / dfov);
        if (circles.find(iphi) == circles.end())
            circles.emplace(iphi, vector<PointXYZ>());
        circles[iphi].push_back(*iter);
    }

    // Find edges
    PointCloud<PointXYZ>::Ptr leftedge(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr rightedge(new PointCloud<PointXYZ>);
    for (auto citer = circles.begin(); citer != circles.end(); citer++)
    {
        float mintheta = numeric_limits<float>::max(), maxtheta = numeric_limits<float>::min();
        for (auto viter = citer->second.begin(); viter != citer->second.end(); viter++)
        {
            float theta = atan2(viter->y, viter->x);
            if (theta < mintheta)
            {
                minpt = *viter;
                mintheta = theta;
            }
            if (theta > maxtheta)
            {
                maxpt = *viter;
                maxtheta = theta;
            }
        }
        leftedge->push_back(minpt);
        rightedge->push_back(maxpt);
    }
    *********************************************/

    // Concave hull
    float nearest_circle_dim = 7; // TODO: adaptive to z
    pcl::PointCloud<pcl::PointXYZ>::Ptr road_hull(new pcl::PointCloud<pcl::PointXYZ>);
    ConcaveHull<PointXYZ> chull;
    chull.setInputCloud(cleanroad);
    chull.setDimension(2);
    chull.setAlpha(nearest_circle_dim);
    chull.reconstruct(*road_hull);
    cerr << " -> Compute concave hull";
#ifdef PROFILE
    cerr << "\tCheckpoint: " << (float)(clock() - t_start) / CLOCKS_PER_SEC << endl;
    t_start = clock();
#endif

    // Split hull
    auto lastpoint = road_hull->back();
    float lastdist = sqrt(lastpoint.x * lastpoint.x + lastpoint.y * lastpoint.y);
    float flip_tolerance = 0.1, dist_tolerance = 0.2;
    bool left = false, lastbigger = true;
    PointCloud<PointXYZ>::Ptr leftedge(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr rightedge(new PointCloud<PointXYZ>);
    CentroidPoint<PointXYZ> leftcenter, rightcenter;
    for (auto viter = road_hull->begin(); viter != road_hull->end(); viter++)
    {
        // float theta = atan2(viter->y, viter->x);
        float dist = sqrt(viter->x * viter->x + viter->y * viter->y);
        if (lastbigger && dist - lastdist > flip_tolerance)
        {
            lastbigger = false;
        }
        if (!lastbigger && lastdist - dist > flip_tolerance)
        {
            left = !left;
            lastbigger = true;
        }

        if (squaredEuclideanDistance(lastpoint, *viter) > dist_tolerance)
        {
            if (left)
            {
                leftedge->push_back(*viter);
                leftcenter.add(*viter);
            }
            else
            {
                rightedge->push_back(*viter);
                rightcenter.add(*viter);
            }
        }

        lastpoint = *viter;
        lastdist = dist;
    }
    cerr << " -> Splited hull";
#ifdef PROFILE
    cerr << "\tCheckpoint: " << (float)(clock() - t_start) / CLOCKS_PER_SEC << endl;
    t_start = clock();
#endif

    if (leftedge->size() == 0 || rightedge->size() == 0)
    {
        cerr << "-> Error when spliting!";
        retcode = 2;
    }
    else
    {
        // Remove unstable end points
        leftedge->erase(leftedge->begin());
        leftedge->erase(leftedge->end() - 1);
        rightedge->erase(rightedge->begin());
        rightedge->erase(rightedge->end() - 1);

        PointXYZ lcpt, rcpt;
        leftcenter.get(lcpt);
        rightcenter.get(rcpt);
        if (lcpt.x > rcpt.x)
            leftedge.swap(rightedge);
    }
    cerr << " -> Hull processed";
#ifdef PROFILE
    cerr << "\tCheckpoint: " << (float)(clock() - t_start) / CLOCKS_PER_SEC << endl;
    t_start = clock();
#endif

    // Display
    if (!embed)
    {
        int port1, port2;
        visualization::PCLVisualizer viewer("Carla Lidar");

        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, port1);
        viewer.setBackgroundColor(0, 0, 0, port1);
        viewer.addPointCloud<pcl::PointXYZ>(leftedge, "left", port1);
        // viewer.addPointCloud<pcl::PointXYZ>(rightedge, "right", port1);
        viewer.addCoordinateSystem(1.0, 2);

        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, port2);
        viewer.setBackgroundColor(0, 0, 0, port2);
        viewer.addPointCloud<pcl::PointXYZ>(cleanroad, "road", port2);
        viewer.addCoordinateSystem(1.0, 2);

        viewer.initCameraParameters();
        viewer.spin();
    }

    // Output
    cerr << " -> Output" << endl;
    if (embed && !retcode)
    {
        cout << leftedge->size() << " " << rightedge->size() << endl;
        for (auto viter = leftedge->begin(); viter != leftedge->end(); viter++)
            cout << viter->x << " " << viter->y << " " << viter->z << endl;
        for (auto viter = rightedge->begin(); viter != rightedge->end(); viter++)
            cout << viter->x << " " << viter->y << " " << viter->z << endl;
    }
#ifdef PROFILE
    cerr << "\tCheckpoint: " << (float)(clock() - t_start) / CLOCKS_PER_SEC << endl;
    t_start = clock();
#endif
}
