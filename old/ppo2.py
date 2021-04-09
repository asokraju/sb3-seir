import time
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

import time
from itertools import permutations
from scipy.io import savemat, loadmat
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import datetime


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

log_dir = "ppo/"
tensorboard_log = "ppo/board/"

# vec_env = make_vec_env(env_id, n_envs=2, wrapper_class=Monitor, monitor_dir=log_dir)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

env_id = 'gym_seir:seir-v0'
env_kwargs = {'validation':False}
env_kwargs['inital_state'] = [99666., 81., 138., 115.]
env = gym.make(env_id,**env_kwargs)
env = Monitor(env, log_dir)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)


# We create a separate environment for evaluation
eval_env = gym.make(env_id,**env_kwargs)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


# Multiprocessed RL Training
n_timesteps = 1e5
start_time = time.time()
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(n_timesteps, tb_log_name="test_1", callback=callback)
total_time_multi = time.time() - start_time
plot_results([log_dir], n_timesteps, results_plotter.X_TIMESTEPS, "SEIR")
plt.show()




####################################################################################################################
# Data Generation
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
    a = model.predict(obs, deterministic=True)[0]
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


####################################################################################################################
##plot

STATES = data['STATES']#[:,:-1]
ACTIONS = data['ACTIONS']
REWARDS = data['REWARDS']
NSTATES = data['NSTATES']

# print(np.shape(STATES))
# print(np.shape(ACTIONS[0]))
col = ['S', 'E', 'I', 'R']
df = pd.DataFrame(STATES, columns=col)
a_map = {0:'LockDown', 1:'Social Distancing', 2:'Open'}
POL = [a_map[a] for a in ACTIONS[0]]
df['A']=POL
pal = {'LockDown':"Red", 'Social Distancing':"Green",'Open':'Blue'}
COSTS = -np.array(REWARDS)
COSTS -= np.mean(COSTS)
COSTS /= (np.std(COSTS) + 1e-10) # normalizing the result
COSTS += 2.
COSTS = COSTS*10.
def pair_plot(df, pal):
    sns.pairplot(df, hue="A", palette=pal)
    plt.savefig("scatter_new_local.pdf", bbox_inches='tight')

pair_plot(df, pal)
df['costs']=COSTS[0]

def plot3d_seir(df, c):
    fig = plt.figure(figsize=(20, 20))
    ax = Axes3D(fig)
    pal = {0:"Red", 1:"Green",2:'Blue'}
    # get colormap from seaborn
    # cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    # colors = sns.color_palette("tab10").as_hex()
    # cmap = ListedColormap([colors[3],colors[0],colors[2]])
    colors = sns.color_palette("Paired").as_hex()
    cmap = ListedColormap([colors[5],colors[1],colors[3]])
    # print([colors[3],colors[0],colors[2]])
    S = df['S']
    E = df['E']
    I = df['I']
    # plot
    sc = ax.scatter(S, E, I, s=df['costs'], c=c, marker='o', alpha=0.6, cmap=cmap)
    ax.set_xlabel('S')
    ax.set_ylabel('E')
    ax.set_zlabel('I')
    x = np.linspace(0., 1e5,1000)
    y = 1e5-x
    zeros = np.zeros(x.shape)
    ax.plot(x,y, 'black', )
    ax.plot(x,zeros, 'black')
    ax.plot(zeros,y, 'black')
    ax.plot(zeros,y,  'black', zs=x)
    ax.plot(zeros,x,  'black', zs=y)
    ax.plot(y, zeros, 'black', zs=x)
    ax.plot(zeros, zeros, 'black', zs=x)
    # legend
    ax.set_xlabel('Susceptible', fontsize=15)
    ax.set_ylabel('Exposed', fontsize=15)
    ax.set_zlabel('Infected', fontsize=15)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 0.9), loc=2)
    # ax.view_init(azim=0, elev=0)
    plt.show()

plot3d_seir(df,ACTIONS)