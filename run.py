import datetime
import os
import numpy as np
import gym
import time

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from itertools import permutations
from scipy.io import savemat, loadmat
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Without this I get some errors/wwarnings. It is only for my local computer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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


def train(w, Senario, args, log_dir):
    env_id = args['env_id']
    n_timesteps = args['n_timesteps']
    check_freq = args['check_freq']
    tensorboard_log = log_dir + "board/"
    env_kwargs = {
        'validation':False,
        'theta':args['theta'][Senario],
        'weight' : w
        }
    env = gym.make(env_id,**env_kwargs)
    env = Monitor(env, log_dir)
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=log_dir)
    model.learn(n_timesteps, tb_log_name="test_1", callback=callback)
    print("Finished training")
    return model

def random_uniform_state():
    popu=1e5
    X = np.random.uniform(low=0.0, high=popu)
    Y = np.random.uniform(low=0.0, high=popu-X)
    Z = np.random.uniform(low=0.0, high=popu-(X + Y))
    W = popu-(X+Y+Z)
    perms = permutations([X, Y, Z, W])
    states = []
    for state in perms:
        states.append(list(state))
    return states

def action_reward(w, log_dir, args, model, inital_state = [99666., 81., 138., 115.]):
    env_id = args['env_id']
    env_kwargs = {
        'validation':True,
        'inital_state' : inital_state,
        'weight' : w
        }
    eval_env = gym.make(env_id,**env_kwargs)
    eval_env = Monitor(eval_env, log_dir)
    obs = eval_env.reset()
    a = model.predict(obs, deterministic=True)[0]
    _,r,_,_ = eval_env.step(a)
    return a, r, eval_env.state

def data_generation(w, log_dir, args, model):
    N = args['N']
    states = []
    Actions = []
    Rewards =[]
    next_states = []
    for _ in range(int(N/16)):
        perms = random_uniform_state()
        for state in perms:
            states.append(list(state))
            a, r, obs = action_reward(w, log_dir, args, model, inital_state = state)
            Actions.append(a)
            Rewards.append(r)
            next_states.append(obs)
    
    data = {
            'STATES' : np.array(states),
            'ACTIONS' : np.array(Actions),
            'REWARDS' : np.array(Rewards),
            'NSTATES' : np.array(next_states),
    }
    file_name = log_dir + 'data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat'
    savemat(file_name, data)
    print("Finished Saved the data at: {}".format(file_name))
    return data

def plot3d_seir(log_dir, df, c):
    fig = plt.figure(figsize=(20, 20))
    ax = Axes3D(fig)
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
    ax.plot(x,y, 'black' )
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
    ax.view_init(azim=40, elev=0)
    # plt.show()
    plt.savefig(log_dir + "3d.jpg", bbox_inches='tight')
    plt.savefig(log_dir + "3d.pdf", bbox_inches='tight')
    plt.close()

def plot_data(args, data, log_dir):
    STATES = data['STATES']#[:,:-1]
    ACTIONS = data['ACTIONS']
    REWARDS = data['REWARDS']
    NSTATES = data['NSTATES']
    col = ['S', 'E', 'I', 'R']
    df = pd.DataFrame(STATES, columns=col)
    a_map = args['a_map']
    POL = [a_map[a] for a in ACTIONS]
    df['A']=POL
    pal = {'LockDown':"Red", 'Social Distancing':"Green",'Open':'Blue'}
    sns.pairplot(df, hue="A", palette=pal, alpha=0.6)
    plt.savefig(log_dir + "scatter_plot.pdf", bbox_inches='tight')
    plt.savefig(log_dir + "scatter_plot.jpg", bbox_inches='tight')
    plt.close()
    COSTS = -np.array(REWARDS)
    COSTS -= np.mean(COSTS)
    COSTS /= (np.std(COSTS) + 1e-10) # normalizing the result
    COSTS += 2.
    COSTS = COSTS*10.
    df['costs']=COSTS[0]
    plot3d_seir(log_dir, df,ACTIONS)



if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    directory = "results/" + start_time + '/'
    try:
        os.mkdir("results/")
        os.mkdir(directory)
    except:
        pass

    args = {
        'n_timesteps' : int(1e5), # No of RL training steps
        'check_freq' : 1000, # frequency of upating the model
        'env_id' : 'gym_seir:seir-v0', # gym environment id
        'N' : 10000, # number of samples to plot
        'theta':{0: 113.92, 1: 87.15, 2: 107.97},
        'w_all' : [0.0 , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
        'sel_w' : [0.4, 0.5, 0.6],
        'Senarios' : [ 'BaseLine', 'Senario_1', 'Senario_2'],
        'a_map' : {0:'LockDown', 1:'Social Distancing', 2:'Open'}
    }
    for w in args['w_all']:
        dir_w = directory + str(w) +"/"
        try:
            os.mkdir(dir_w)
        except:
            pass
        
        for i, senario in enumerate(args['Senarios']):
            dir_sen = dir_w + senario + "/"
            try:
                os.mkdir(dir_sen)
            except:
                pass
            print(dir_sen)
            start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            
            model = train(w, i, args, log_dir=dir_sen)
            data = data_generation(w, dir_sen, args, model)
            plot_data(args, data, dir_sen)

            end_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            print(start_time, end_time)


