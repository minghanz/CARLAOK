#include <detector_src.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <string>

void detect(const PointCloudType &cloud, std::vector<Detection3D> &dobjects, std::vector<PointCloudType::Ptr> &oclouds,
    float cluster_thres, float ransac_thres)
{
    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<PointType> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.05);

    PointCloudType::Ptr cloud_filtered (new PointCloudType(cloud)), cloud_f (new PointCloudType);
    int i=0, nr_points = (int) cloud.points.size ();
    while (true)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<PointType> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;

        // Judge the points associated with the planar surface
        std::cout << "PointCloud representing the planar component: " << inliers->indices.size ()
                  << " data points." << std::endl;
        if(inliers->indices.size () < ransac_thres * nr_points) break;
    }
    std::cout << "PointCloud representing the objects: " << cloud_filtered->size ()
              << " data points." << std::endl;

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance (cluster_thres); // m
    ec.setMinClusterSize (10);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);
    std::cout << "Objects: " << cluster_indices.size () << std::endl;

    int j = 0; Detection3D dobj;
    // pcl::visualization::PCLVisualizer vis;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        PointCloudType::Ptr cloud_cluster (new PointCloudType);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        // estimate(cloud_cluster, dobj);
        // dobjects.push_back(dobj);
        oclouds.push_back(cloud_cluster);

        // pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(cloud_cluster, i%256, (i*i)%256, (i*i*i)%256);
        // vis.addPointCloud(cloud_cluster, single_color, "cloud"+std::to_string(j));
        // vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud"+std::to_string(j));

        j++;
    }
    // vis.spin();
}

// void estimate(const PointCloudType::Ptr object, Detection3D &bound)
// {    
//     Eigen::Vector4f centroid;
//     pcl::compute3DCentroid (*object, centroid);
//     bound.x = centroid(0);
//     bound.y = centroid(1);
//     bound.z = centroid(2);
    
//     Eigen::MatrixXf mat = object->getMatrixXfMap();
//     Eigen::MatrixXf xy = mat.leftCols(2);
//     xy.col(0).array() -= centroid(0);
//     xy.col(1).array() -= centroid(1);
//     Eigen::JacobiSVD<Eigen::MatrixXf> svd(xy, Eigen::ComputeFullV);
//     Eigen::Matrix2f v = svd.matrixV().transpose();
//     bound.l = (xy*v.col(0)).maxCoeff() - (xy*v.col(0)).minCoeff();
//     bound.w = (xy*v.col(1)).maxCoeff() - (xy*v.col(1)).minCoeff();
//     bound.h = mat.col(2).maxCoeff() - mat.col(2).minCoeff();
// }

