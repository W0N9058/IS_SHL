import os
import time
import argparse
import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from message.msg import Result, Query
from rccar_gym.env_wrapper import RCCarWrapper

###################################################
########## YOU CAN ONLY CHANGE THIS PART  #########

"""
Freely import modules, define methods and classes, etc.
You may add other python codes, but make sure to push it to github.
To use particular modules, please let TA know to install them on the evaluation server, too.
If you want to use a deep-learning library, please use pytorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

TEAM_NAME = "SHL"

# DDPG Actor, Critic 정의
class DDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DDPGCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

###################################################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)") # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)") # or ../..
    parser.add_argument("--render", default=True, action='store_true', help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action='store_true', help="No rendering.")
    parser.add_argument("--mode", default='val', type=str, help="Whether train new model or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'")
    parser.add_argument("--model_dir", default="model", type=str, help="Model path relative to 'IS_TEAMNAME/project/'")
    
    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
    """
    Change the model name as you want.
    Note that this will used for evaluation by the server as well.
    You can add any arguments you want.
    """
    parser.add_argument("--model_name", default="model.pkl", type=str, help="model name to save and use")
    ###################################################
    ###################################################
    
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    # render
    if args.no_render:
        args.render = False
    
    ws_path = os.path.join(get_package_prefix('rccar_bringup'), "../..")

    # map files
    args.maps = os.path.join(ws_path, 'maps')
    args.maps = [map for map in os.listdir(args.maps) if os.path.isdir(os.path.join(args.maps, map))]
    
    # configuration files
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)

    with open(args.env_config, 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open(args.dynamic_config, 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(task_args)
    args.update(dynamic_args)
    
    # Trajectory & Model Path
    project_path = os.path.join(ws_path, f"src/rccar_bringup/rccar_bringup/project/IS_{TEAM_NAME}/project")
    args.traj_dir = os.path.join(project_path, args.traj_dir)
    args.model_dir = os.path.join(project_path, args.model_dir)
    args.model_path = os.path.join(args.model_dir, args.model_name)

    return args


class RCCarPolicy(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_project3")
        self.args = args
        self.mode = args.mode

        self.query_sub = self.create_subscription(Query, "/query", self.query_callback, 10)
        self.result_pub = self.create_publisher(Result, "/result", 10)

        self.dt = args.timestep
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer
        self.maps = args.maps 
        self.render = args.render
        self.time_limit = 180.0

        self.traj_dir = args.traj_dir
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.model_path = args.model_path

        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)

        # DDPG를 위한 설정
        self.obs_dim = 720  # scan dimension
        self.act_dim = 2   # [steer, speed]

        # Actor, Critic 정의
        self.actor = DDPGActor(self.obs_dim, self.act_dim)
        self.critic = DDPGCritic(self.obs_dim, self.act_dim)

        # Actor 학습용 옵티마이저 (demonstration 기반 imitation)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # 초기값 (train 시 사용)
        self.obs_mean = None
        self.obs_std = None
        self.act_mean = None
        self.act_std = None

        self.load()
        self.get_logger().info(">>> Running Project 3 for TEAM {}".format(TEAM_NAME))
        ###################################################
        ###################################################
        
    def train(self):
        self.get_logger().info(">>> Start model training")
        # demonstration data를 불러와서 actor를 모사학습
        obs_data_path = os.path.join(self.traj_dir, "obs_map2.npy")
        act_data_path = os.path.join(self.traj_dir, "act_map2.npy")

        if not os.path.exists(obs_data_path) or not os.path.exists(act_data_path):
            raise FileNotFoundError(f"Training data not found in {self.traj_dir}.")

        obs_data = np.load(obs_data_path)
        act_data = np.load(act_data_path)

        # Normalize
        self.obs_mean, self.obs_std = obs_data.mean(axis=0), obs_data.std(axis=0)
        self.act_mean, self.act_std = act_data.mean(axis=0), act_data.std(axis=0)

        self.obs_std[self.obs_std == 0] = 1e-4
        self.act_std[self.act_std == 0] = 1e-4

        obs_data_norm = (obs_data - self.obs_mean) / self.obs_std
        act_data_norm = (act_data - self.act_mean) / self.act_std

        obs_tensor = torch.tensor(obs_data_norm, dtype=torch.float32)
        act_tensor = torch.tensor(act_data_norm, dtype=torch.float32)

        num_epochs = 500
        batch_size = 128
        best_loss = float('inf')
        best_model_path = os.path.join(self.model_dir, "best_ddpg_model.pkl")

        for epoch in range(num_epochs):
            indices = np.arange(obs_tensor.size(0))
            np.random.shuffle(indices)

            epoch_loss = 0.0
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                obs_batch = obs_tensor[batch_indices]
                act_batch = act_tensor[batch_indices]

                pred_act = self.actor(obs_batch)
                loss = ((pred_act - act_batch)**2).mean()

                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

                epoch_loss += loss.item()
            epoch_loss /= (len(indices)/batch_size)
            self.get_logger().info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.actor.state_dict(), best_model_path)
                self.get_logger().info(f"New Best Model Saved at {best_model_path} Loss: {best_loss}")

        # 최종 모델 저장
        torch.save(self.actor.state_dict(), self.model_path)
        np.save(os.path.join(self.model_dir, "obs_mean_2.npy"), self.obs_mean)
        np.save(os.path.join(self.model_dir, "obs_std_2.npy"), self.obs_std)
        np.save(os.path.join(self.model_dir, "act_mean_2.npy"), self.act_mean)
        np.save(os.path.join(self.model_dir, "act_std_2.npy"), self.act_std)
        self.get_logger().info(">>> Trained model {} is saved".format(self.model_name))
            
    def load(self):
        if self.mode == 'val':
            # best_ddpg_model.pkl을 로드하도록 수정
            best_model_path = os.path.join(self.model_dir, "best_ddpg_model.pkl")
            assert os.path.exists(best_model_path), f"Best model not found at {best_model_path}"
            self.actor.load_state_dict(torch.load(best_model_path))
            self.obs_mean = np.load(os.path.join(self.model_dir, "obs_mean_2.npy"))
            self.obs_std = np.load(os.path.join(self.model_dir, "obs_std_2.npy"))
            self.act_mean = np.load(os.path.join(self.model_dir, "act_mean_2.npy"))
            self.act_std = np.load(os.path.join(self.model_dir, "act_std_2.npy"))
        elif self.mode == 'train':
            pass
        else:
            raise AssertionError("mode should be one of 'train' or 'val'.")   

    def get_action(self, obs):
        # obs를 정규화하고 actor로부터 deterministic action 산출
        if self.obs_mean is None or self.obs_std is None:
            # 모델이 로드되지 않은 경우 기본값 사용
            # 여기서는 편의상 zero mean, one std 가정
            self.obs_mean = np.zeros(self.obs_dim)
            self.obs_std = np.ones(self.obs_dim)
            self.act_mean = np.zeros(self.act_dim)
            self.act_std = np.ones(self.act_dim)

        obs = torch.tensor((obs - self.obs_mean)/self.obs_std, dtype=torch.float32)
        with torch.no_grad():
            action_norm = self.actor(obs)
            action = action_norm * torch.tensor(self.act_std, dtype=torch.float32) + torch.tensor(self.act_mean, dtype=torch.float32)
            steer = np.clip(action[0].item(), -self.max_steer, self.max_steer)
            speed = np.clip(action[1].item(), self.min_speed, self.max_speed)
        
        return np.array([[steer, speed]])
    

    def query_callback(self, query_msg):
        
        id = query_msg.id
        team = query_msg.team
        map = query_msg.map
        trial = query_msg.trial
        exit = query_msg.exit 
        
        result_msg = Result()
        
        START_TIME = time.time()
            
        try:
            if team != TEAM_NAME:
                return
            
            if map not in self.maps:
                END_TIME = time.time()
                result_msg.id = id
                result_msg.team = team
                result_msg.map = map
                result_msg.trial = trial
                result_msg.time = END_TIME - START_TIME
                result_msg.waypoint = 0
                result_msg.n_waypoints = 20
                result_msg.success = False
                result_msg.fail_type = "Invalid Track"
                self.get_logger().info(">>> Invalid Track")
                self.result_pub.publish(result_msg)
                return
            
            self.get_logger().info(f"[{team}] START TO EVALUATE! MAP NAME: {map}")
            
            ### New environment
            env = RCCarWrapper(args=self.args, maps=[map], render_mode="human_fast" if self.render else None)
            track = env._env.unwrapped.track
            if self.render:
                env.unwrapped.add_render_callback(track.centerline.render_waypoints)
            
            obs, _ = env.reset(seed=self.args.seed)
            _, _, scan = obs

            step = 0
            terminate = False

            while True:  
                act = self.get_action(scan)
                steer = np.clip(act[0][0], -self.max_steer, self.max_steer)
                speed = np.clip(act[0][1], self.min_speed, self.max_speed)
                
                obs, _, terminate, _, info = env.step(np.array([steer, speed]))
                _, _, scan = obs
                step += 1
                
                if self.render:
                    env.render()

                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt
                    result_msg.waypoint = info['waypoint']
                    result_msg.n_waypoints = 20
                    result_msg.success = False
                    result_msg.fail_type = "Time Out"
                    self.get_logger().info(">>> Time Out: {}".format(map))
                    self.result_pub.publish(result_msg)
                    env.close()
                    break

                if terminate:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt 
                    result_msg.waypoint = info['waypoint']
                    result_msg.n_waypoints = 20
                    if info['waypoint'] == 20:
                        result_msg.success = True
                        result_msg.fail_type = "-"
                        self.get_logger().info(">>> Success: {}".format(map))
                    else:
                        result_msg.success = False
                        result_msg.fail_type = "Collision"
                        self.get_logger().info(">>> Collision: {}".format(map))
                    self.result_pub.publish(result_msg)
                    env.close()
                    break
        except Exception as e:
            END_TIME = time.time()
            result_msg.id = id
            result_msg.team = team
            result_msg.map = map
            result_msg.trial = trial
            result_msg.time = END_TIME - START_TIME
            result_msg.waypoint = 0
            result_msg.n_waypoints = 20
            result_msg.success = False
            result_msg.fail_type = "Script Error"
            self.get_logger().info(f">>> Script Error: {e}")
            self.result_pub.publish(result_msg)
        
        if exit:
            rclpy.shutdown()
        return

def main():
    args = get_args()
    rclpy.init()
    node = RCCarPolicy(args)
    if args.mode == 'train':
        node.train()
    else:
        rclpy.spin(node)


if __name__ == '__main__':
    main()
