#Pure_pursuit订阅的节点：
Final\_waypoints（由之前几个节点对waypoints文件处理之后的最终waypoints）

Current\_pose（无人车当前相对于地图的坐标，来自ndt_matching）

Current\_velocity（无人车当前的估计速率，来自ndt_matching）

Config/waypoint_follower（来自Runtime Manager的该节点APP配置）

#Pure_pursuit的主要功能：
根据目前的车辆位置，从Final\_waypoints一系列节点中的某一个（或2个），生成目标位置（target)。根据车辆当前位置和目标位置生成一个curvature(路径曲线），并计算出相应的转角，以Final\_waypoints列表的第一个waypoint的速度作为规划速度，结合这些生成twist，发布给twist\_filter，以控制车辆最终的方向。

#Pure_pursuit主循环结构：
在该Node的函数PurePursuitNode::run（）的主循环中，每次循环都等待重新获取的（Final\_waypoints + Current\_pose + Current\_velocity），依据最新的这三组数据，计算出新的twist，并进行发布，同时发布可视化的信息，用于在rviz中显示。 然后将这三组数据丢弃，等待新的数据到来，进入下一次的计算。

#每次的计算过程：
调用 setLookaheadDistance(computeLookaheadDistance());
 setMinimumLookaheadDistance(minimum\_lookahead\_distance\_);
调用canGetCurvature（这里是最关键的算法）进行计算。

其中LookaheadDistance = 10 * current\_linear\_velocity,这个数值来自该节点所订阅的/Current\_velocity中的x线速度。

Ld = current\_linear\_velocity * lookahead\_distance\_ratio(=2). 
实际是取ld，并给它以maximum和minimum上下限进行限制。 

显然，maximum是不可能超出的。
在做好设置之后，就进入了核心代码：canGetCurvature(pure\_pursuit.cpp)。

#canGetCurvature架构：
#有必要显示出在程序运行期间，pure\_pursuit所获得的lane的样子。
* (1).调用getNextWaypoint从lane当中选择一个合适的waypoint作为接下来进行跟踪的waypoint.
* (2).在当前lane中枚举所有waypoint，只要有一个waypoint的pose（位置）与车辆的pose(位置）距离之差大于minimum\_lookahead\_distance，那么就认为当前正在跟踪的lane是合理的，否则直接返回。
* (3).判断当前是否支持差值模式，或选择的NextWaypoint是当前lane的第一个或最后一
个点，则直接以NextWaypoint为目标计算Curvature并返回。
* (4).若nextWaypoint是lane的中间一个waypoint，并且支持差值模式，进入差值模式，生成一个新的目标点，并计算当前位置到该位置的curvature。 

#GetNextwaypoint原理：
从lane的第一个waypoint往后，计算每一个waypoint位置与无人车当前位置的距离，若距离大于lookahead\_distance\_，则返回。也就是从lane中找到第一个距离无人车位置“足够远”的waypoint，选择它为接下来无人车行动的目标点。
Interpolate差值算法：
当nextWaypoint既不是当前lane的开头，也不是结尾，并且支持差值模式的情况下，以差值的方式创建一个目标点。

* (1).选择nextWaypoint的前一个waypoint的作为差值算法的start，将nextWaypoint看做差值算法的end。
* (2).在start点和end点之间画一条直线。
* (3).计算无人车当前位置到该直线的距离（垂线长度），若距离太远（大于lookahead\_distance），则放弃差值，失败返回。
* (4).计算出无人车当前位置到该直线的投影垂足(h1,h2，哪个距离直线更近选哪个,以ERROR为上限，都超出ERROR，则失败返回）。
* (5).以lookahead\_distance为半径，以无人车当前位置为圆心做圆。注意无人车与start和end之间的直线距离小于等于lookahead\_distance。因此有两种情况：该圆与该直线有1个交点（相切）或有2个交点。
* (6).1个交点，将垂足作为目标点返回
* (7).2个交点：记lookahead\_distance为l，无人车与start和end之间的直线距离为d。计算：。通过s，计算出5所说的圆与直线的两个交点，检查哪个交点在start和end之间，返回该点为无人车的下一个目标。

以上。

可以看到，pure\_pursuit从它所获取到的/Final\_waypoints这个waypoint的集合中，根据种种条件以两个waypoint差值算出合适的目标（以差值算法为主）。并根据该目标规划无人车接下来的动作Twist。  

#计算Curvature(kappa)的方法
NewPos为目标点以车辆目前位置和朝向进行坐标转换后，相对于无人车坐标系的新坐标。
Double kappa = NewPos.y*2 / distance(target,current\_pose)



#由curvature计算最终Twist的方法
Twist.linear.x = 取当前lane第一个waypoint的线速度;
Twist.angular.z = kappa * Twist.linear.x; 



#finally:
一个重点是： pure\_pursuit在规划下一个目标点的时候，的确没有考虑waypoint的朝向（orientation），而仅仅考虑了waypoint的position。并且以该lane的第一个waypoint指定的线速度作为规划的线速度，而没有考虑整条lane其他waypoint的线速度。

关于线速度和角度速以及curvature的数学计算公式等等数学问题，留待以后慢慢解决。
