import gym
import numpy as np
import concurrent.futures 
import time
import matplotlib.pyplot as plt
import copy
import os

from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from scipy.io import savemat
from itertools import permutations
from stable_baselines3.common.monitor import Monitor
import datetime

path = r"C:\Users\kkris\Documents\GitHub\sb3-seir\ppo\best_model.zip"
load_model = PPO.load(path)
log_dir = "ppo/"

def random_uniform_state():
    popu=1e5
    S = np.random.uniform(low=0.0, high=popu)
    E = np.random.uniform(low=0.0, high=popu-S)
    I = np.random.uniform(low=0.0, high=popu-(S+E))
    R = popu-(S+E+I)
    state = [S,E,I,R]
    perms = permutations([S,E,I,R])
    states = []
    for state in perms:
        states.append(state)
    return states

def action_reward(inital_state = [99666., 81., 138., 115.]):
    env_id = 'gym_seir:seir-v0'
    env_kwargs = {'validation':True}
    env_kwargs['inital_state'] = inital_state
    eval_env = gym.make(env_id,**env_kwargs)
    eval_env = Monitor(eval_env, log_dir)
    obs = eval_env.reset()
    a = load_model.predict(obs, deterministic=True)[0]
    _,r,_,_ = eval_env.step(a)
    return a, r, eval_env.state

def random_uniform_state():
    popu=1e5
    S = np.random.uniform(low=0.0, high=popu)
    E = np.random.uniform(low=0.0, high=popu-S)
    I = np.random.uniform(low=0.0, high=popu-(S+E))
    R = popu-(S+E+I)
    return [S,E,I,R]
states = []
Actions = []
Rewards =[]
N =10000
next_states = []

for _ in range(int(N/16)):
    print("{}/1000".format(_))
    perms = permutations(random_uniform_state())
    for state in perms:
        states.append(list(state))
        a, r, obs = action_reward(list(state))
        Actions.append(a)
        Rewards.append(r)
        next_states.append(obs)
    
data = {
        'STATES' : np.array(states),
        'ACTIONS' : np.array(Actions),
        'REWARDS' : np.array(Rewards),
        'NSTATES' : np.array(next_states),
}
file_name = 'data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat'
savemat(file_name, data)