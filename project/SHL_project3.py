import os
import time
import argparse
import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict
import pickle

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from message.msg import Result, Query
from rccar_gym.env_wrapper import RCCarWrapper

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random

###################################################
########## YOU CAN ONLY CHANGE THIS PART ##########

"""
Freely import modules, define methods and classes, etc.
You may add other python codes, but make sure to push it to github.
To use particular modules, please let TA know to install them on the evaluation server, too.
If you want to use a deep-learning library, please use pytorch.
"""

TEAM_NAME = "SHL"

###################################################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)")
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)")
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
    parser.add_argument("--model_name", default="last_model.pkl", type=str, help="Model name to save and use")
    ###################################################
    ###################################################
    
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPOPolicy, self).__init__()
        self.policy_mean = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )
        self.policy_std = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim),
            nn.Softplus()
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        mean = self.policy_mean(obs)
        std = self.policy_std(obs)
        return mean, std, self.value_net(obs)


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
        """
        Freely change the codes to increase the performance.
        """
        
        self.obs_dim = 720  # Assuming 720-dimensional LiDAR scan
        self.act_dim = 2    # Assuming 2 action dimensions: steer and speed
        self.policy = PPOPolicy(self.obs_dim, self.act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4, weight_decay=1e-5)
        #self.optimizer = optim.AdamW(self.policy.parameters(), lr=3e-4, weight_decay=1e-4)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)
        self.mse_loss = nn.MSELoss()

        self.load()
        self.get_logger().info(">>> Running Project 3 for TEAM {}".format(TEAM_NAME))
        self.prev_act_batch = None
    
    def train(self):
        self.get_logger().info(">>> Start model training")
        """
        Train and save your model.
        You can either use this part or explicitly train using other python codes.
        """
        
        # Load training data
        obs_data_path = os.path.join(self.traj_dir, "obs_map5.npy")
        act_data_path = os.path.join(self.traj_dir, "act_map5.npy")

        if not os.path.exists(obs_data_path) or not os.path.exists(act_data_path):
            raise FileNotFoundError(f"Training data not found in {self.traj_dir}.")

        obs_data = np.load(obs_data_path)
        act_data = np.load(act_data_path)

        self.get_logger().info(f"Loaded training data: {obs_data.shape[0]} samples.")

        # Normalize the data
        self.obs_mean, self.obs_std = obs_data.mean(axis=0), obs_data.std(axis=0)
        self.act_mean, self.act_std = act_data.mean(axis=0), act_data.std(axis=0)

        self.obs_std[self.obs_std == 0] = 1e-4
        self.act_std[self.act_std == 0] = 1e-4
    
        obs_data = (obs_data - self.obs_mean) / self.obs_std
        act_data = (act_data - self.act_mean) / self.act_std

        obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
        act_tensor = torch.tensor(act_data, dtype=torch.float32)

        # Training loop
        num_epochs = 200
        batch_size = 128
        best_loss = float('inf')  # Initialize to a large value
        best_model_path = os.path.join(self.model_dir, "best_model.pkl")

        for epoch in range(num_epochs):
            np.random.seed(self.args.seed + epoch)
            indices = np.arange(obs_data.shape[0])
            np.random.shuffle(indices)

            epoch_loss = 0  # To track total loss for this epoch
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                obs_batch = obs_tensor[batch_indices]
                act_batch = act_tensor[batch_indices]

                mean, std, _ = self.policy(obs_batch)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(act_batch).abs()
                
                log_probs_loss = log_probs.sum(dim=-1).mean()
                if self.prev_act_batch is not None:
                    min_size = min(act_batch.size(0), self.prev_act_batch.size(0))
                    steer_change = act_batch[:min_size, 0] - self.prev_act_batch[:min_size, 0] 
                    steer_penalty = (steer_change ** 2).mean()
                else:
                    steer_penalty = 0.0
                lambda_penalty = 0.3
                loss = log_probs_loss + lambda_penalty * steer_penalty
                
                #loss = log_probs.sum(dim=-1).mean()
                
                #steer_weight = torch.abs(act_batch[:, 0]) / self.max_steer
                #weighted_log_probs = log_probs[:, 0] * steer_weight + log_probs[:, 1]
                #loss = weighted_log_probs.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
                #self.prev_act_batch = act_batch

                epoch_loss += loss.item()  # Accumulate loss for the epoch

            epoch_loss /= (len(indices) / batch_size)  # Average loss for the epoch
            self.get_logger().info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

            # Save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.policy.state_dict(), best_model_path)
                self.get_logger().info(f">>> New best model saved with loss: {best_loss}")

        # Save the last model
        torch.save(self.policy.state_dict(), self.model_path)
        self.get_logger().info(f">>> Last model saved as {self.model_path}")
        
        np.save(os.path.join(self.model_dir, "obs_mean.npy"), self.obs_mean)
        np.save(os.path.join(self.model_dir, "obs_std.npy"), self.obs_std)
        np.save(os.path.join(self.model_dir, "act_mean.npy"), self.act_mean)
        np.save(os.path.join(self.model_dir, "act_std.npy"), self.act_std)

    def load(self):
        """
        Load your trained model.
        Make sure not to train a new model when self.mode == 'val'.
        """
        if self.mode == 'val':
            assert os.path.exists(self.model_path)
            self.policy.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.obs_mean = np.load(os.path.join(self.model_dir, "obs_mean.npy"))
            self.obs_std = np.load(os.path.join(self.model_dir, "obs_std.npy"))
            self.act_mean = np.load(os.path.join(self.model_dir, "act_mean.npy"))
            self.act_std = np.load(os.path.join(self.model_dir, "act_std.npy"))
        elif self.mode == 'train':
            pass
        else:
            raise ValueError("Mode should be 'train' or 'val'.")

    def get_action(self, obs):
        """
        Predict action using obs - 'scan' data.
        Be sure to satisfy the limitation of steer and speed values.
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            obs = (obs - self.obs_mean) / self.obs_std
            mean, std, _ = self.policy(obs)
            #dist = Normal(mean, std)
            #sampled_action = dist.sample()
            #action = sampled_action * self.act_std + self.act_mean
            action = mean * self.act_std + self.act_mean
            steer = np.clip(action[0].item(), -self.max_steer, self.max_steer)
            speed = self.max_speed
        return np.array([steer, speed])
        
    ###################################################
    ###################################################

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
                obs, _, terminate, _, info = env.step(act)
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
