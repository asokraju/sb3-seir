import datetime
import os
from os import walk, listdir

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
import plotly
import plotly.graph_objs as go 
import plotly.express as px
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

def action_reward(w, log_dir, args, model, inital_state):
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

def GenerateStates(N=int(1e4)):
    States = []
    popu=1e5
    for _ in range(N):
        S = np.random.uniform(low=0.0, high=popu)
        E = np.random.uniform(low=0.0, high=popu-S)
        I = np.random.uniform(low=0.0, high=popu-(S+E))
        R = popu-(S+E+I)
        state = [S,E,I,R]
        perms = permutations([S,E,I,R])
        for p in perms:
            States.append(list(p))
    return States


def DataGeneration(w, log_dir, args, model, States):
    N = args['N']
    Actions = []
    Rewards =[]
    next_states = []
    env_id = args['env_id']
    l = len(States)
    for i, state in enumerate(States):
        percent  = (i/l)*100
        print("\r", "{}/{:.3f}".format(i,percent), end="")
        env_kwargs = {
            'validation':True,
            'inital_state' : state,
            'weight' : w
            }
        eval_env = gym.make(env_id,**env_kwargs)
        eval_env = Monitor(eval_env, log_dir)
        obs = eval_env.reset()
        a = model.predict(obs, deterministic=True)[0]
        _,r,_,_ = eval_env.step(a)
        a, r, eval_env.state = action_reward(w, log_dir, args, model, inital_state = state)
        Actions.append(a)
        Rewards.append(r)
        next_states.append(obs)
    
    data = {
            'STATES' : np.array(States),
            'ACTIONS' : np.array(Actions),
            'REWARDS' : np.array(Rewards),
            'NSTATES' : np.array(next_states),
    }
    file_name = log_dir + 'data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat'
    savemat(file_name, data)
    print("Finished Saved the data at: {}".format(file_name))
    return data

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
    S = df['Susceptible']
    E = df['Exposed']
    I = df['Infected']
    # plot
    sc = ax.scatter(S, E, I, s=df['costs'], c=c, marker='o', alpha=0.6, cmap=cmap)
    ax.set_xlabel('Susceptible')
    ax.set_ylabel('Exposed')
    ax.set_zlabel('Infected')
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
    plt.savefig(log_dir + "3d_new.jpg", bbox_inches='tight')
    # plt.savefig(log_dir + "3d_new.pdf", bbox_inches='tight')
    plt.close()

def plot_data(args, data, log_dir, title):
    STATES = data['STATES']#[:,:-1]
    ACTIONS = data['ACTIONS']
    if np.shape(ACTIONS)[0]==1:
        ACTIONS = ACTIONS[0]
    REWARDS = data['REWARDS']
    NSTATES = data['NSTATES']
    col = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    df = pd.DataFrame(STATES, columns=col)
    a_map = args['a_map']
    POL = [a_map[a] for a in ACTIONS]
    df['Policy']=POL
    plot_kws={'alpha': 0.6}
    hue_order = ['LockDown','Social Distancing', 'Open']
    pal = {'LockDown':"Red", 'Social Distancing':"Green",'Open':'Blue'}
    # g = sns.pairplot(df, kind='scatter', alpha=0.1})
    g = sns.pairplot(df, hue="Policy",  palette=pal, plot_kws = plot_kws, hue_order = hue_order)
    # g.set_title(title)
    # g.title(title)
    # g.savefig(log_dir + "scatter_plot_new.pdf", bbox_inches='tight')
    g.savefig(log_dir + "scatter_plot_new7.jpg", bbox_inches='tight')
    # g.close()

    COSTS = -np.array(REWARDS)
    COSTS -= np.mean(COSTS)
    COSTS /= (np.std(COSTS) + 1e-10) # normalizing the result
    COSTS += 2.
    COSTS = COSTS*10.
    df['costs']=COSTS[0]
    plot3d_seir(log_dir, df,ACTIONS)

def Trajectories(w, log_dir, args, model, inital_state, Senario):
    env_id = args['env_id']
    done = False
    env_kwargs = {
        'validation':True,
        'inital_state' : inital_state,
        'weight' : w
    }
    done = False
    eval_env = gym.make(env_id,**env_kwargs)
    eval_env = Monitor(eval_env, log_dir)
    obs = eval_env.reset()
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs,r,done,_ = eval_env.step(a)
    States = eval_env.state_trajectory[:-1]
    Actions = eval_env.action_trajectory
    # Rewards = eval_env.rewards
    # print(eval_env.rewards, eval_env.weekly_rewards)
    WeeklyRewards = eval_env.weekly_rewards
    Rewards = [r for r in WeeklyRewards for _ in range(eval_env.time_steps)]
    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    col = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    df = pd.DataFrame(States, columns=col, index = index)
    df['Policy'] = Actions
    df['Rewards'] = Rewards
    title_sen = ['BaseLine', 'First', 'Second']
    fig = go.Figure([{
        'x': index,#S.index,
        'y': df[col],
        'name': col
    }  for col in ['Susceptible', 'Exposed', 'Infected', 'Recovered']])
    fig.update_layout(
        title= "States, " + "%s - Senario"%title_sen[Senario] + ". w = %s"%w,
        xaxis_title="Time - months",
        yaxis_title="No. of people",
        legend_title="States",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
    )
    # fig.show()
    fig.write_image(log_dir + "States.pdf")
    fig.write_image(log_dir + "States.jpeg")
    fig.write_image(log_dir + "States.png")
    plotly.offline.plot(fig, filename=log_dir + 'States.html')

    fig = go.Figure([{
        'x': index,#S.index,
        'y': df.Rewards,
        'name': 'Rewards'
    } ])
    fig.update_layout(
        title="Rewards," + " %s - Senario"%title_sen[Senario] + ". w = %s"%w,
        xaxis_title="Time - months",
        yaxis_title="Rewards",
        # legend_title="States",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
    )
    fig.write_image(log_dir + "Rewards.pdf")
    fig.write_image(log_dir + "Rewards.jpeg")
    fig.write_image(log_dir + "Rewards.png")
    plotly.offline.plot(fig, filename='Actions.html')

    fig = go.Figure([{
        'x': index,#S.index,
        'y': df.Policy,
        'name': 'Policy'
    } ])
    fig.update_layout(
        title="Policy: 0-Open, 1-Social Distancing, 2-Lockdown." + " %s - Senario"%title_sen[Senario] + ". w = %s"%w,
        xaxis_title="Time - months",
        yaxis_title="Actions",
        yaxis_range=[-0.2, 2.2],
        # legend_title="States",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
    )
    fig.write_image(log_dir + "Actions.pdf")
    fig.write_image(log_dir + "Actions.jpeg")
    fig.write_image(log_dir + "Actions.png")
    plotly.offline.plot(fig, filename=log_dir + 'Rewards.html')  



def Trajectories_comparision(w, log_dir, args, model, inital_state):
    env_id = args['env_id']
    done = False
    env_kwargs = {
        'validation':True,
        'inital_state' : inital_state,
        'weight' : w
    }
    done = False
    eval_env = gym.make(env_id,**env_kwargs)
    eval_env = Monitor(eval_env, log_dir)
    obs = eval_env.reset()
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs,r,done,_ = eval_env.step(a)
    States = eval_env.state_trajectory[:-1]
    Actions = eval_env.action_trajectory
    # Rewards = eval_env.rewards
    # print(eval_env.rewards, eval_env.weekly_rewards)
    WeeklyRewards = eval_env.weekly_rewards
    Rewards = [r for r in WeeklyRewards for _ in range(eval_env.time_steps)]
    index = pd.date_range("2020-05-15 00:00:00", "2020-11-05 23:55:00", freq="5min")
    col = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
    df = pd.DataFrame(States, columns=col, index = index)
    df['Policy'] = Actions
    df['Rewards'] = Rewards

    fig = go.Figure([{
        'x': index,#S.index,
        'y': df[col],
        'name': col
    }  for col in ['Susceptible', 'Exposed', 'Infected', 'Recovered']])
    fig.update_layout(
        title="w = %s"%w,
        xaxis_title="Time - months",
        yaxis_title="No. of people",
        legend_title="States",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
    )
    # fig.show()
    fig.write_image(log_dir + "States.pdf")
    fig.write_image(log_dir + "States.jpeg")
    fig.write_image(log_dir + "States.png")
    plotly.offline.plot(fig, filename=log_dir + 'States.html')

    fig = go.Figure([{
        'x': index,#S.index,
        'y': df.Rewards,
        'name': 'Rewards'
    } ])
    fig.update_layout(
        title="Rewards",
        xaxis_title="Time - months",
        yaxis_title="Rewards",
        # legend_title="States",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
    )
    fig.write_image(log_dir + "Actions.pdf")
    fig.write_image(log_dir + "Actions.jpeg")
    fig.write_image(log_dir + "Actions.png")
    plotly.offline.plot(fig, filename='Actions.html')

    fig = go.Figure([{
        'x': index,#S.index,
        'y': df.Policy,
        'name': 'Policy'
    } ])
    fig.update_layout(
        title="0-Open, 1-Social Distancing, 2-Lockdown",
        xaxis_title="Time - months",
        yaxis_title="Actions",
        yaxis_range=[0, 2],
        # legend_title="States",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="RebeccaPurple"
        )
    )
    fig.write_image(log_dir + "Rewards.pdf")
    fig.write_image(log_dir + "Rewards.jpeg")
    fig.write_image(log_dir + "Rewards.png")
    plotly.offline.plot(fig, filename=log_dir + 'Rewards.html')  

if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    directory = "results/" + "21-04-09-16-55" + '/'
    try:
        os.mkdir("results/")
    except:
        pass
    try:
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
        'a_map' : {0:'LockDown', 1:'Social Distancing', 2:'Open'},
        'initial_state':{
            0:[99666., 81., 138., 115.], 
            1:[99962.0, 7.0, 14.0, 17.0],
            2:[99905.0, 22.0, 39.0, 34.0]
            }
    }
    States = GenerateStates(N=int(args['N']/16))
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

            # training the model
            # model = train(w, i, args, log_dir=dir_sen)

            # Load the trained model
            model = PPO.load(dir_sen+ "best_model.zip")

            # Generate the data
            # data = DataGeneration(w, dir_sen, args, model, States)
            # data = data_generation(w, dir_sen, args, model)

            # load the generated data
            # for f in listdir(dir_sen):
            #     if f.endswith('.' + 'mat'):
            #         mat_file_names = dir_sen+ "/" + f
            # print(mat_file_names)
            # data = loadmat(mat_file_names)

            # # ploting the data
            plots_dir = dir_sen + "plots-" + "w=%s"%w +"-%s"%senario + "/"
            try:
                os.mkdir(plots_dir)
            except:
                pass
            # if i==0:
            #     title = "w={}".format(w) + ", Baseline"
            # elif i==1:
            #     title = "w={}".format(w) + ", Senario - 1"
            # elif i==2:
            #     title = "w={}".format(w) + ", Senario - 2"
            # plot_data(args, data, dir_sen, title=title)
            Trajectories(w, plots_dir, args, model, inital_state=args['initial_state'][i], Senario=i)
            # end_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            # print(start_time, end_time)


