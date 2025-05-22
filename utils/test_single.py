import numpy as np
from utils.arguments import get_args
from rl_modules.SAC_agent import Agent
import torch
from environments.environment import SatelliteEnvApp
import random
from utils.utils import resource_path


def get_env_params(argus, env):
    params = {'obs': 3,  # observation dim
              'goal': 3,  # goal dim
              'action': 48*env.satellite_number+1,  # action dim, 1代表全局不动
              'done': 1,
              'sat_num': env.satellite_number,
              'max_timestep': argus.max_cycle_steps
              }
    return params


def launch(argus, module_num, initial_config, target_config):
    # create the sac_agent
    env = SatelliteEnvApp(argus.seed, module_num, initial_config, target_config)
    # set random seeds for reproduce
    if argus.seed is not None:
        random.seed(argus.seed)
        np.random.seed(argus.seed)
        torch.manual_seed(argus.seed)
    # get the environment parameters
    env_params = get_env_params(argus, env)
    # create the ddpg agent to interact with the environment
    sac_tester = Agent(argus, env, env_params)
    if module_num == 4:
        sac_tester.loadCheckpoints(resource_path("model/agent_4_sat"))
    else:
        sac_tester.loadCheckpoints(resource_path("model/agent_6_sat"))
    sac_tester.test_cycle_app()


def run(module_num, initial_config, target_config):
    # get the params
    args = get_args()
    if module_num == 4:
        args.max_cycle_steps = 16
    else:
        args.max_cycle_steps = 25

    launch(args, module_num, initial_config, target_config)
