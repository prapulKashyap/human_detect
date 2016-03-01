#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/Vertices.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/impl/conditional_euclidean_clustering.hpp>

#include <pcl/filters/extract_indices.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/surface/mls.h>
#include <pcl/surface/concave_hull.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <boost/thread/thread.hpp>

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/people/ground_based_people_detection_app.h>

ros::Publisher pub;
ros::Publisher pub_obj;
//ros::Publisher vis_pub;

typedef pcl::PointXYZRGB PointT;

float deg2rad(float alpha)
{
  return (alpha * 0.017453293f);
}




void mark_cluster(pcl::PointCloud<PointT>::Ptr cloud_cluster ,int id)
{
  Eigen::Vector4f centroid;
  Eigen::Vector4f min;
  Eigen::Vector4f max;
 
  pcl::compute3DCentroid (*cloud_cluster, centroid);
  pcl::getMinMax3D (*cloud_cluster, min, max);
 
  uint32_t shape = visualization_msgs::Marker::CUBE;
  visualization_msgs::Marker marker;
  marker.header.frame_id = cloud_cluster->header.frame_id;
  marker.header.stamp = ros::Time::now();
 
  marker.ns = "people_found";
  marker.id = id;
  marker.type = shape;
  marker.action = visualization_msgs::Marker::ADD;
 
  marker.pose.position.x = centroid[0];
  marker.pose.position.y = centroid[1];
  marker.pose.position.z = centroid[2];
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
 
  marker.scale.x = (max[0]-min[0]);
  marker.scale.y = (max[1]-min[1]);
  marker.scale.z = (max[2]-min[2]);
 
  if (marker.scale.x ==0)
      marker.scale.x=0.1;

  if (marker.scale.y ==0)
    marker.scale.y=0.1;

  if (marker.scale.z ==0)
    marker.scale.z=0.1;
   
  marker.color.r = 0;
  marker.color.g = 255;
  marker.color.b = 0;
  marker.color.a = 0.5;

  marker.lifetime = ros::Duration();
//   marker.lifetime = ros::Duration(0.5);
  vis_pub.publish( marker );
  //return marker;
} 






void ransac(const sensor_msgs::PointCloud2ConstPtr& input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected)
{
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr init_cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr people_cloud(new pcl::PointCloud<PointT>);

  pcl::fromROSMsg(*input, *cloud);
  *init_cloud = *cloud;

  //*cloud_projected = *cloud;

  
  pcl::ModelCoefficients::Ptr floor_coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr floor_indices(new pcl::PointIndices());
  pcl::SACSegmentation<PointT> floor_finder;
  floor_finder.setOptimizeCoefficients(true);
  floor_finder.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
  // floor_finder.setModelType (SACMODEL_PLANE);
  floor_finder.setMethodType(pcl::SAC_RANSAC);
  floor_finder.setMaxIterations(300);
  floor_finder.setAxis(Eigen::Vector3f(0, 0, 1));
  floor_finder.setDistanceThreshold(0.008);
  floor_finder.setEpsAngle(deg2rad(5));
  floor_finder.setInputCloud(boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >(*cloud));
  floor_finder.segment(*floor_indices, *floor_coefficients);

  if (floor_indices->indices.size() > 0)
  {
    // Extract the floor plane inliers
    pcl::PointCloud<PointT>::Ptr floor_points(new pcl::PointCloud<PointT>);
    pcl::ExtractIndices<PointT> extractor;
    extractor.setInputCloud(boost::make_shared<pcl::PointCloud<PointT> >(*cloud));
    extractor.setIndices(floor_indices);
    extractor.filter(*floor_points);
    extractor.setNegative(true);
    extractor.filter(*cloud);

    // Project the floor inliers
    pcl::ProjectInliers<PointT> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(floor_points);
    proj.setModelCoefficients(floor_coefficients);
    proj.filter(*cloud_projected);
    floor_points->header.frame_id = "camera_link";
    floor_points->header.stamp = ros::Time::now().toNSec();
   }
   
   ground_cloud->width  = 3;
   ground_cloud->height = 1;
   ground_cloud->points.resize (ground_cloud->width * ground_cloud->height);

   ground_cloud->points[0] = cloud_projected->points[ rand()%cloud_projected->points.size()  ];
   ground_cloud->points[1] = cloud_projected->points[ rand()%cloud_projected->points.size()  ];
   ground_cloud->points[2] = cloud_projected->points[ rand()%cloud_projected->points.size()  ];
   
  float min_confidence = -1.5;
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.06;
  Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics
   Eigen::VectorXf ground_coeffs;
   ground_coeffs.resize(4);
   std::vector<int> ground_points_indices;
  for (unsigned int i = 0; i < ground_cloud->points.size(); i++)
    ground_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(ground_cloud);
  model_plane.computeModelCoefficients(ground_points_indices,ground_coeffs);


  // Create classifier for people detection:  
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile("/home/busygoose/human_detect_ws/src/pcl_495/src/trainedLinearSVMForPeopleDetectionWithHOG.yaml");   // load trained SVM



  // People detection app initialization:
  pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setHeightLimits(min_height, max_height);         // set person classifier

  // Perform people detection on the new cloud:
  std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
  people_detector.setInputCloud(init_cloud);
  people_detector.setGround(ground_coeffs);                    // set floor coefficients
  people_detector.compute(clusters);                           // perform people detection

  
  ground_coeffs = people_detector.getGround();                 // get updated floor coefficients
  int people_found=0;
  std::vector< int > people_indices;

  for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
      {
        if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
        {
          
          std::vector< int > people_index = it->getIndices().indices;
          for(int p=0;p<people_index.size();p++)
          {
             people_indices.push_back(people_index.at(p));

          }
          
          people_found++;
        }
      }

if(people_found>0)
{

  
  for(int l=0;l<people_indices.size();l++)
  {
     people_cloud->push_back(init_cloud->points[people_indices.at(l)]);

  }
  
}
if(people_found>0)
{
    std::cerr<<"people found "<<people_found<<std::endl;
    //mark_cluster(people_cloud ,0);
    *cloud_projected = *init_cloud;
}



}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Do data processing here...
  // run ransac to find floor
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZRGB>);
  ransac(input, cloud_projected);
  pub.publish(*cloud_projected);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_node_w_nodelets");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
  //vis_pub = nh.advertise<visualization_msgs::Marker>( "visualization_marker", 0 );
  pub_obj = nh.advertise<sensor_msgs::PointCloud2> ("objects", 1);

  // Spin
  ros::spin ();
}

