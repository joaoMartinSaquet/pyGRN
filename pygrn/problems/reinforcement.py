from .base import Problem
import numpy as np
from loguru import logger
import gymnasium as gym

ENV_POOL = []



def create_env_info(env):

    env_info = {}
    env_info["action_space"] = env.action_space
    env_info["observation_space"] = env.observation_space

    return env_info

def set_env_pool(envs):
    global ENV_POOL
    ENV_POOL = envs



class ReinforcementLearningTask(Problem):


    def __init__(self, env_info, env_name="Pendulum-v1", env = None):
        


        namestr = env_name
        super().__init__(namestr)

        self.env = None
        if env is not None:
            self.env = env
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.has_continuous_action = isinstance(action_space, gym.spaces.Box)
            self.has_continuous_observation = isinstance(observation_space, gym.spaces.Box)
            self.nin = self.env.observation_space.shape[0]

            logger.warning(" using a single environment") 

        else:
            action_space = env_info["action_space"]
            observation_space = env_info["observation_space"]
            self.has_continuous_action = isinstance(action_space, gym.spaces.Box)
            self.has_continuous_observation = isinstance(observation_space, gym.spaces.Box)
            self.nin = observation_space.shape[0]
        
        self.dtype = float
        if self.has_continuous_action:
            self.nout = action_space.shape[0]
            self.h_act = action_space.high
            self.l_act = action_space.low
        else:
            self.nout = 1
            self.n = action_space.n
            self.dtype = int
        
        self.cacheable = True



    def eval(self, grn):
        fit = -100 # to ensure that even if not evaluated it is not selected
        reward = 0
        ts = 0
        seed = np.random.randint(0, 5)

        if self.env is None:
            while len(ENV_POOL) < 1:
                pass
            eval_env = ENV_POOL.pop()
        else:
            eval_env = self.env

        grn.setup()
        grn.warmup(25)
        seed = np.random.randint(0, 5)
        obs, _ = eval_env.reset(seed=seed)
        done = False

        while True:


            # if self.has_continuous_observation:
            #     (obs - self.l_obs) / (self.h_obs - self.l_obs)  # map action to [0, 1]
            grn.set_input(obs)
            grn.step()
            # print("grn best output ", grn.get_output().item())
            if self.has_continuous_action:
                action = grn.get_output() * (self.h_act - self.l_act) + self.l_act
            else:
                action = int(grn.get_output()[0] * (self.n - 1))

            # print("action ", action)s
            # print(action)
            # action = 1 if grn.get_output().item() > 0.5 else 0
            obs, r, terminated, truncated, info = eval_env.step(action)
            
            if terminated or truncated:
                done = True
                break
            reward += r
            ts += 1
  
            fit = reward

            # if fit ==0:
                # logger.debug(f"fit = {fit} ")
        if self.env is None:
            ENV_POOL.append(eval_env)
        # print("reward = ", reward, "\t steps = ", ts)
        return fit 