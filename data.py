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
from plotly.subplots import make_subplots
from plotly.offline import iplot
from sklearn.metrics import confusion_matrix
from pandas.plotting import table 

# Without this I get some errors/wwarnings. It is only for my local computer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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

w=0.5

path = r"C:\Users\kkris\Documents\GitHub\sb3-seir\results\21-04-09-16-55\0.5\actions.csv"
df = pd.read_csv(path)

Misfits_1 = df[df['BaseLine']!=df['Senario_1']]
Misfits_2 = df[df['BaseLine']!=df['Senario_2']]
Misfits_2.sort_values(by='Susceptible', ascending=False, inplace=True)
Misfits_1.sort_values(by='Susceptible', ascending=False, inplace=True)

print(df.head())
print(Misfits_1.head())
print(Misfits_2.head())

def plot3d_seir(log_dir, df, c):
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)
    colors = sns.color_palette("Paired").as_hex()
    cmap = ListedColormap([colors[3],colors[5],colors[1]])
    # print([colors[3],colors[0],colors[2]])
    S = df['Susceptible']
    E = df['Exposed']
    I = df['Infected']
    # plot
    sc = ax.scatter(S, E, I, s=20, c=c, marker='o', alpha=0.6, cmap=cmap)
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
    ax.view_init(azim=120, elev=10)
    # plt.show()
    plt.savefig(log_dir + "3d_new.jpg", bbox_inches='tight')
    # plt.savefig(log_dir + "3d_new.pdf", bbox_inches='tight')
    plt.close()

def plot_data(args, df, log_dir, fig_name):
    ACTIONS = df['BaseLine']
    col = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Policy']
    # df = pd.DataFrame(STATES, columns=col)
    a_map = args['a_map']
    POL = [a_map[a] for a in ACTIONS]
    df['Policy']=POL
    plot_kws={'alpha': 0.6}
    hue_order = ['LockDown','Social Distancing', 'Open']
    pal = {'LockDown':"Red", 'Social Distancing':"Green",'Open':'Blue'}
    # # g = sns.pairplot(df, kind='scatter', alpha=0.1})
    # g = sns.pairplot(df[col], hue="Policy",  palette=pal, plot_kws = plot_kws, hue_order = hue_order)
    # # g.set_title(title)
    # # g.title(title)
    # # g.savefig(log_dir + "scatter_plot_new.pdf", bbox_inches='tight')
    # g.savefig(log_dir + fig_name, bbox_inches='tight')
    # # g.close()
    plot3d_seir(log_dir, df,ACTIONS)
# log_dir = r"C:\Users\kkris\Documents\GitHub\sb3-seir\results\21-04-09-16-55\0.5" + "\\"
# plot_data(args, Misfits_1, log_dir, fig_name='scatter_residual_1.jpg')
# plot_data(args, Misfits_2, log_dir, fig_name='scatter_residual_2.jpg')
# print(Misfits_1.describe())
# print(Misfits_2.describe())

# # ax = plt.subplot(111, frame_on=False) # no visible frame
# # ax.xaxis.set_visible(False)  # hide the x axis
# # ax.yaxis.set_visible(False)  # hide the y axis
# des = Misfits_1.describe()
# print(des.index)
# col = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
# idx = ['mean', 'std', 'min', 'max']
# des = des.loc[idx]
# # table(ax, Misfits_1.describe())  # where df is your data frame
# fig, ax = plt.subplots(figsize=(8,5)) 
# sns.heatmap(des[col],annot=True, vmin=0., vmax=1e5, fmt='g')
# plt.savefig(log_dir + 'mytable1.png')

# des = Misfits_2.describe()
# des = des.loc[idx]
# # table(ax, Misfits_1.describe())  # where df is your data frame
# fig, ax = plt.subplots(figsize=(8,5)) 
# sns.heatmap(des[col],annot=True, vmin=0., vmax=1e5, fmt='g')
# plt.savefig(log_dir + 'mytable2.png')
# ax = plt.subplot(111, frame_on=False) # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis

# table(ax, Misfits_2.describe())  # where df is your data frame

# plt.savefig(log_dir + 'mytable2.png')
# dir_sen = dir_w + args['Senarios'][0] + "/"
# # load the generated data
# for f in listdir(dir_sen):
#     if f.endswith('.' + 'mat'):
#         mat_file_names = dir_sen+ "/" + f
# print(mat_file_names)
# data = loadmat(mat_file_names)
# STATES = data['STATES']#[:,:-1]
# ACTIONS = data['ACTIONS']
# if np.shape(ACTIONS)[0]==1:
#     ACTIONS = ACTIONS[0]
# col = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
# df = pd.DataFrame(STATES, columns=col)
# df['Actions'] = ACTIONS
# print(df.head(),np.shape(ACTIONS))

# df_1 = df[df['Actions']==0]
# print(df_1.head())
