import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from adk_node.msg import WaypointPath
from adk_node.msg import TargetPerception

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from math import sin,cos,pi
from tf_transformations import quaternion_from_euler 
from tf_transformations import euler_from_quaternion 
from tf_transformations import euler_from_matrix

from utils.utility_functions import *
from mission import *


# Run the labyrinth navigation experiment.
''' Conventions

Directions:
----------
0 = right
1 = down
2 = left
3 = up

Actions:
--------
left = 0
right = 1
forward = 2
'''


# %%
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('..')

from Environments.minigrid_labyrinth import Maze
import numpy as np
from Controllers.minigrid_controller import MiniGridController
from Controllers.meta_controller import MetaController
import pickle
import os, sys
from datetime import datetime
from MDP.high_level_mdp import HLMDP
from utils.results_saver import Results
import time

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('sparse_waypoint_publisher')
        qos_profile = QoSProfile(depth=1,reliability=QoSReliabilityPolicy.RELIABLE,durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher_ = self.create_publisher(WaypointPath, 'adk_node/input/waypoints', qos_profile)
        self.create_subscription(TargetPerception, 'adk_node/ground_truth/perception', self.gt_perception_callback, 10)
        self.create_subscription(Odometry, 'adk_node/SimpleFlight/odom_local_ned', self.odom_callback, 10)
        self.odom_msg = None
        self.detected_entity_ids = []
        self.visited_cell_idxs = []

        self.setup()

        while self.odom_msg == None:
            rclpy.spin_once(self)

        self.get_logger().info('Odom Message Received...')

        self.mission = Mission('../../../mission_briefing/description.json')
        
        self.run_mission()

    def run_mission(self):
        for car in self.mission.car_list:
            print(car.id)
            print('priority: ', car.priority)
            self.run_single_eoi_search(car)

    def run_single_eoi_search(self, car):
        for region in car.map:
            print('probability: ',region.probability)
            print(region.polygon)
            print('cells: ', region.cells)
            for cell_idx in region.cells:
                if cell_idx in self.visited_cell_idxs: continue
                hl_controller_idx_list = self.dummy_hl_controller(cell_idx)
                hl_controller_list = self.get_controller_list(hl_controller_idx_list)
                self.run_minigrid_solver_and_pub(hl_controller_list)
                current_minigrid_state = airsim2minigrid((self.n_airsim, self.e_airsim, 0))
                final_minigrid_state = hl_controller_list[-1].get_final_states()
                while (current_minigrid_state != final_minigrid_state):
                    rclpy.spin_once(self)
                    current_minigrid_state = airsim2minigrid((self.n_airsim, self.e_airsim, 0))
                    final_minigrid_state = list(hl_controller_list[-1].get_final_states()[0])
                    print("Current State: {}, Final State: {}".format(current_minigrid_state, final_minigrid_state))   
                    if car.id in self.detected_entity_ids: return


    def get_controller_list(self,hl_controller_idx_list):
        controller_list = []
        for hl_controller_id in hl_controller_idx_list:
            for controller in self.controller_list:
                if controller.controller_ind == hl_controller_id:
                    controller_list.append(controller)
        return controller_list

    def dummy_hl_controller(self, cell_idx):
        controller_list = []
        if cell_idx == 0:
            controller_list.extend([3,1])
        if cell_idx == 1:
            controller_list.extend([0,2])
        if cell_idx == 2:
            controller_list.extend([3,4])
        if cell_idx == 3:
            controller_list.append(7)
        if cell_idx == 5:
            controller_list.extend([8,12,11])
        if cell_idx == 6:
            controller_list.extend([10,13])
        if cell_idx == 7:
            controller_list.extend([12,14])
        if cell_idx == 9:
            controller_list.extend([15,18])
        if cell_idx == 10:
            controller_list.extend([22,21])
        if cell_idx == 14:
            controller_list.append(29)
        return controller_list

    def setup(self):
        # %% Setup and create the environment
        env_settings = {
            'agent_start_states' : [(22,2,0)], # Need to obtain this from start position -> config file
            'slip_p' : 0.0,
        }

        self.env = Maze(**env_settings)

        # env.render(highlight=False)
        # time.sleep(5)


        num_rollouts = 5
        meta_controller_n_steps_per_rollout = 500

        # %% Set the load directory (if loading pre-trained sub-systems) or create a new directory in which to save results

        # load_folder_name = '2023-10-12_13-13-22_minigrid_labyrinth'
        load_folder_name = '2024-04-01_17-26-42_minigrid_labyrinth'
        save_learned_controllers = True

        experiment_name = 'minigrid_labyrinth'

        base_path = os.path.abspath(os.path.curdir)
        string_ind = base_path.find('src')
        assert(string_ind >= 0)
        base_path = base_path[0:string_ind + 4]
        base_path = os.path.join(base_path, 'verifiable-compositional-rl/src/data', 'saved_controllers')

        load_dir = os.path.join(base_path, load_folder_name)

        # %% Load the sub-system controllers
        self.controller_list = []
        for controller_dir in os.listdir(load_dir):
            controller_load_path = os.path.join(load_dir, controller_dir)
            if os.path.isdir(controller_load_path):
                controller = MiniGridController(0, load_dir=controller_load_path)
                self.controller_list.append(controller)
        
        self.obs = self.env.reset() # Get the first state

    def gt_perception_callback(self, gt_perception_msg):
        entity_status = "entered" if gt_perception_msg.enter_or_leave == 0 else "left"
        entity_id = gt_perception_msg.entity_id
        self.get_logger().info('Entity {} just {}...'.format(entity_id, entity_status))
        if entity_id not in self.detected_entity_ids: self.detected_entity_ids.append(entity_id)

    def odom_callback(self, odom_msg):
        self.odom_msg = odom_msg
        self.n_airsim = odom_msg.pose.pose.position.x
        self.e_airsim = odom_msg.pose.pose.position.y
        print("Current AirSim State: {}".format([self.n_airsim, self.e_airsim, 0]))

    def pub_waypoint(self, obs_list):
        waypoint_msg = WaypointPath()
        
        pose_msg_array = []

        for airsim_obs in obs_list:
            z = -10.0
            n_airsim, e_airsim, yaw_airsim = airsim_obs

            roll = 0
            pitch = 0
            yaw = 0

            q=quaternion_from_euler(roll*pi/180, pitch*pi/180, yaw*pi/180)


            pose_msg = PoseStamped()

            h = Header()
            h.stamp = self.get_clock().now().to_msg()

            pose_msg.header = h
            pose_msg.header.frame_id = "world_ned"
            pose_msg.pose.position.x = n_airsim
            pose_msg.pose.position.y = e_airsim
            pose_msg.pose.position.z = z

            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]

            pose_msg_array.append(pose_msg)

        waypoint_msg.path = pose_msg_array
        waypoint_msg.velocity = 8.0
        waypoint_msg.lookahead = -1.0
        waypoint_msg.adaptive_lookahead = 0.0
        # waypoint_msg.drive_train_type = "ForwardOnly"
        waypoint_msg.wait_on_last_task = False

        self.publisher_.publish(waypoint_msg)
        self.get_logger().info('Publishing Waypoints...')
    
    def run_minigrid_solver_and_pub(self,hl_controller_list):
        obs_list = []
        print("Init State: "+str(self.obs))
        for controller in hl_controller_list:
            init = True
            if init:
                print("Final State: "+str(controller.get_final_states())+"\n")
                print("** Using Controller **: "+str(controller.controller_ind)+"\n")
                init = False
            while (self.obs != controller.get_final_states()).any():
                action,_states = controller.predict(self.obs, deterministic=True)
                self.obs, reward, done, info = self.env.step(action)
                print("Action: "+str(action))
                print("Current State: "+str(self.obs))
                # env.render(highlight=False)
                # time.sleep(0.5)
                # AirSim mapping
                airsim_obs = minigrid2airsim(self.obs)
                print("AirSim State: "+str(airsim_obs)+"\n")
                obs_list.append(airsim_obs)
        self.pub_waypoint(obs_list)



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
