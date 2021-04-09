import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import os
from scipy.io import savemat, loadmat
import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# file ="data_21-04-08-06-09.mat"
file = "data_21-04-08-17-42.mat"
file = "data_21-04-08-17-40.mat"
file = 'data_21-04-09-15-16.mat'
file = r'C:\Users\kkris\Documents\GitHub\sb3-seir\plots\data_21-04-09-15-16.mat'
file = r'C:\Users\kkris\Documents\GitHub\sb3-seir\data_21-04-09-10-46.mat'
data = loadmat(file)

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

