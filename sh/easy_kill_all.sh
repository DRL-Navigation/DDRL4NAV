# only used in local, for easily debug code.
ps -ef|grep main.py|grep -v grep|awk '{print $2}'|xargs kill -9
ps -ef|grep gazebo|grep -v grep|awk '{print $2}'|xargs kill -9
#ps -ef|grep robot_state_publisher|grep -v grep|awk '{print $2}'|xargs kill -9
#ps -ef|grep collision_publisher_node|grep -v grep|awk '{print $2}'|xargs kill -9
#ps -ef|grep ros|grep -v grep|awk '{print $2}'|xargs kill -9

