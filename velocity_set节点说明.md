## velocity_set节点说明

### 订阅的topic

```
 // velocity set subscriber
  ros::Subscriber waypoints_sub = nh.subscribe("safety_waypoints", 1, &VelocitySetPath::waypointsCallback, &vs_path);//obstacle_avoid发送的topic
  ros::Subscriber current_vel_sub =
      nh.subscribe("current_velocity", 1, &VelocitySetPath::currentVelocityCallback, &vs_path);//vel_ray发送的topic，车辆当前速度

  // velocity set info subscriber
  ros::Subscriber config_sub = nh.subscribe("config/velocity_set", 1, &VelocitySetInfo::configCallback, &vs_info);//配置信息
  ros::Subscriber points_sub = nh.subscribe(points_topic, 1, &VelocitySetInfo::pointsCallback, &vs_info);//经过处理后的激光雷达点云数据
  ros::Subscriber localizer_sub = nh.subscribe("localizer_pose", 1, &VelocitySetInfo::localizerPoseCallback, &vs_info);//传感器姿态
  ros::Subscriber control_pose_sub = nh.subscribe("current_pose", 1, &VelocitySetInfo::controlPoseCallback, &vs_info);//车辆当前姿态
  ros::Subscriber obstacle_sim_points_sub = nh.subscribe("obstacle_sim_pointcloud", 1, &VelocitySetInfo::obstacleSimCallback, &vs_info);//obstacle_sim产生的点云数据
  ros::Subscriber detectionresult_sub = nh.subscribe("/state/stopline_wpidx", 1, &VelocitySetInfo::detectionCallback, &vs_info);

  // vector map subscriber
  ros::Subscriber sub_dtlane = nh.subscribe("vector_map_info/cross_walk", 1, &CrossWalk::crossWalkCallback, &crosswalk);
  ros::Subscriber sub_area = nh.subscribe("vector_map_info/area", 1, &CrossWalk::areaCallback, &crosswalk);
  ros::Subscriber sub_line = nh.subscribe("vector_map_info/line", 1, &CrossWalk::lineCallback, &crosswalk);
  ros::Subscriber sub_point = nh.subscribe("vector_map_info/point", 1, &CrossWalk::pointCallback, &crosswalk);
```

### 发送的topic

```

```



主要流程

```
ros::Rate loop_rate(LOOP_RATE);
  while (ros::ok())
  {
    ros::spinOnce();

    int closest_waypoint = 0;
	//加载crosswalk数据
    if (crosswalk.loaded_all && !crosswalk.set_points)
      crosswalk.setCrossWalkPoints();

    if (!vs_info.getSetPose() || !vs_path.getSetPath())
    {
      loop_rate.sleep();
      continue;
    }

    crosswalk.setMultipleDetectionFlag(enable_multiple_crosswalk_detection);

    if (use_crosswalk_detection)
      crosswalk.setDetectionWaypoint(
          crosswalk.findClosestCrosswalk(closest_waypoint, vs_path.getPrevWaypoints(), STOP_SEARCH_DISTANCE));

    int obstacle_waypoint = -1;
    //检测障碍物，并返回状态结果
    EControl detection_result = obstacleDetection(closest_waypoint, vs_path.getPrevWaypoints(), crosswalk, vs_info,detection_range_pub, obstacle_pub, &obstacle_waypoint);
	//对waypoint根据状态结果进行更新
    changeWaypoints(vs_info, detection_result, closest_waypoint,
                    obstacle_waypoint, final_waypoints_pub, &vs_path);

    vs_info.clearPoints();

    // publish obstacle waypoint index
    std_msgs::Int32 obstacle_waypoint_index;
    obstacle_waypoint_index.data = obstacle_waypoint;
    obstacle_waypoint_pub.publish(obstacle_waypoint_index);

    vs_path.resetFlag();

    loop_rate.sleep();
  }
```

### 主要功能函数

障碍物检测的主要原理：

对最近的waypoint之后一定范围的waypoint进行遍历，然后遍历点云中的每个点到waypoint的距离是否在一定范围内，若在则进行保存，遍历减速后判断保存的障碍物点数量是否大于阀值，若大于则认为在该waypoint点上有障碍物。

停止障碍物检测

```
int detectStopObstacle(const pcl::PointCloud<pcl::PointXYZ>& points, const int closest_waypoint,
                       const autoware_msgs::lane& lane, const CrossWalk& crosswalk, double stop_range,
                       double points_threshold, const geometry_msgs::PoseStamped& localizer_pose,
                       ObstaclePoints* obstacle_points, EObstacleType* obstacle_type,
                       const int wpidx_detection_result_by_other_nodes)
{
  int stop_obstacle_waypoint = -1;
  *obstacle_type = EObstacleType::NONE;
  // start search from the closest waypoint
  for (int i = closest_waypoint; i < closest_waypoint + STOP_SEARCH_DISTANCE; i++)
  {
    // reach the end of waypoints
    if (i >= static_cast<int>(lane.waypoints.size()))
      break;

    // detection another nodes
    if (wpidx_detection_result_by_other_nodes >= 0 &&
        lane.waypoints.at(i).gid == wpidx_detection_result_by_other_nodes)
    {
      stop_obstacle_waypoint = i;
      *obstacle_type = EObstacleType::STOPLINE;
      obstacle_points->setStopPoint(lane.waypoints.at(i).pose.pose.position); // for vizuialization
      break;
    }

    // Detection for cross walk
    if (i == crosswalk.getDetectionWaypoint())
    {
      // found an obstacle in the cross walk
      if (crossWalkDetection(points, crosswalk, localizer_pose, points_threshold, obstacle_points) == EControl::STOP)
      {
        stop_obstacle_waypoint = i;
        *obstacle_type = EObstacleType::ON_CROSSWALK;
        break;
      }
    }
    //计算全局坐标系相对于机器人坐标系中的位置,即point相对于pose坐标系的位置
    // waypoint seen by localizer
    geometry_msgs::Point waypoint = calcRelativeCoordinate(lane.waypoints[i].pose.pose.position, localizer_pose.pose);
    //将point转换为Vector3
    tf::Vector3 tf_waypoint = point2vector(waypoint);
    tf_waypoint.setZ(0);

    int stop_point_count = 0;
    for (const auto& p : points)
    {
      tf::Vector3 point_vector(p.x, p.y, 0);
      //当障碍物点到waypoint距离小于stop_range，则stop_point_count增加，并将点保存在obstacle_points中
      // 2D distance between waypoint and points (obstacle)
      double dt = tf::tfDistance(point_vector, tf_waypoint);
      if (dt < stop_range)
      {
        stop_point_count++;
        geometry_msgs::Point point_temp;
        point_temp.x = p.x;
        point_temp.y = p.y;
        point_temp.z = p.z;
        obstacle_points->setStopPoint(calcAbsoluteCoordinate(point_temp, localizer_pose.pose));
      }
    }
    //当stop_point_count大于阀值，就会返回当前waypoint，并将obstacle_type设置为ON_WAYPOINTS
    // there is an obstacle if the number of points exceeded the threshold
    if (stop_point_count > points_threshold)
    {
      stop_obstacle_waypoint = i;
      *obstacle_type = EObstacleType::ON_WAYPOINTS;
      break;
    }

    obstacle_points->clearStopPoints();

    // check next waypoint...
  }

  return stop_obstacle_waypoint;
}
```

减速障碍物检测

```

```

waypoint速度的更改主要是通过公式v = (v0)^2 + 2ax来进行计算

更改waypoint以达到停止的目的

```
void VelocitySetPath::changeWaypointsForStopping(int stop_waypoint, int obstacle_waypoint, int closest_waypoint, double deceleration)
{
  if (closest_waypoint < 0)
    return;

  // decelerate with constant deceleration
  for (int index = stop_waypoint; index >= closest_waypoint; index--)
  {
    if (!checkWaypoint(index, __FUNCTION__))
      continue;
    //计算index点的速度
    // v = (v0)^2 + 2ax, and v0 = 0
    double changed_vel = std::sqrt(2.0 * deceleration * calcInterval(index, stop_waypoint));

    double prev_vel = prev_waypoints_.waypoints[index].twist.twist.linear.x;
    if (changed_vel > prev_vel)
    {
      new_waypoints_.waypoints[index].twist.twist.linear.x = prev_vel;
    }
    else
    {
      new_waypoints_.waypoints[index].twist.twist.linear.x = changed_vel;
    }
  }

  // fill velocity with 0 for stopping
  for (int i = stop_waypoint; i <= obstacle_waypoint; i++)
  {
    new_waypoints_.waypoints[i].twist.twist.linear.x = 0;
  }

}
```

更改waypoint以达到减速的目的

```
void VelocitySetPath::changeWaypointsForDeceleration(double deceleration, int closest_waypoint, int obstacle_waypoint)
{
  double square_vel_min = decelerate_vel_min_ * decelerate_vel_min_;
  int extra = 4; // for safety
  // decelerate with constant deceleration
  for (int index = obstacle_waypoint + extra; index >= closest_waypoint; index--)
  {
    if (!checkWaypoint(index, __FUNCTION__))
      continue;

    // v = sqrt( (v0)^2 + 2ax )
    double changed_vel = std::sqrt(square_vel_min + 2.0 * deceleration * calcInterval(index, obstacle_waypoint));

    double prev_vel = prev_waypoints_.waypoints[index].twist.twist.linear.x;
    if (changed_vel > prev_vel)
    {
      new_waypoints_.waypoints[index].twist.twist.linear.x = prev_vel;
    }
    else
    {
      new_waypoints_.waypoints[index].twist.twist.linear.x = changed_vel;
    }
  }

}
```

