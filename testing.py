import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


os.environ['KMP_DUPLICATE_LIB_OK']='True'
env = gym.make('gym_seir:seir-v0')

done = False
while not done:
    s,r,done,_ = env.step(env.action_space.sample())

plt.plot(env.state_trajectory)
plt.show()
check_env(env)
