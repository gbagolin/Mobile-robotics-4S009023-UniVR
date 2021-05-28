"""Launch file for the discrete DDQN algorithm

This script instantiate the gym environment, the agent, and start the training
"""

import argparse
import os

import gym
import yaml
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment

from agent import DDQN
from utils.tracker import Tracker

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

if not cfg['setup']['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Gym env', default=cfg['train']['name'])
parser.add_argument('-epochs', type=int, help='Epochs', default=cfg['train']['n_episodes'])
parser.add_argument('-verbose', type=int, help='Save stats freq', default=cfg['train']['verbose'])
parser.add_argument('-eps_d', type=float, help='Exploration decay', default=cfg['agent']['eps_d'])
parser.add_argument('-tau', type=float, help='Target net τ', default=cfg['agent']['tau'])
parser.add_argument('-use_polyak', type=float, help='Polyak update', default=cfg['agent']['polyak'])
parser.add_argument('-tg_update', type=float, help='Standard update', default=cfg['agent']['tg_update'])

def main(params):
    config = vars(parser.parse_args())

    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(file_name=None, side_channels=[channel])
    channel.set_configuration_parameters(time_scale = 20.0)

    env = UnityToGymWrapper(unity_env)
    
    agent = DDQN(env, cfg['agent'])
    tag = 'DDQN'

    # Initiate the tracker for stats
    tracker = Tracker(
        "TurtleBot3",
        tag,
        seed,
        cfg['agent'], 
        ['Epoch', 'Ep_Reward']
    )

    # Train the agent
    agent.train(
        tracker,
        n_episodes=config['epochs'], 
        verbose=config['verbose'],
        params=cfg['agent'],
        hyperp=config
    )

if __name__ == "__main__":
    main(cfg)
