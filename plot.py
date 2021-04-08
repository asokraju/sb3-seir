import gym
import numpy as np
import matplotlib.pyplot as plt

import os

from scipy.io import savemat, loadmat

import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

file ="data_21-04-08-06-09.mat"
data = loadmat(file)

STATES = data['STATES']
ACTIONS = data['ACTIONS']
REWARDS = data['REWARDS']
NSTATES = data['NSTATES']

# generate data
n = 200
S = STATES[:,0]
E = STATES[:,1]
I = STATES[:,2]
COSTS = -np.array(REWARDS)
COSTS -= np.mean(COSTS)
COSTS /= (np.std(COSTS) + 1e-10) # normalizing the result
COSTS = COSTS*20
# axes instance
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plot
sc = ax.scatter(S, E, I, s=5, c=ACTIONS, marker='o', alpha=1)
ax.set_xlabel('S')
ax.set_ylabel('E')
ax.set_zlabel('I')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
ax.view_init(azim=0, elev=0)
plt.show()
# rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
# save
# plt.savefig("scatter_hue", bbox_inches='tight')
a_map = {0:'O', 1:'S', 2:'C'}
# POL = [a_map[a] for a in ACTIONS[0].tolist()]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
ax.scatter(S, E, c=ACTIONS, s=60, alpha=1, edgecolors='none')
ax.set_ylabel('S', fontsize=40)
ax.set_xlabel('E', fontsize=40)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_ylim(0,1e5)
ax.set_xlim(0,1e5)
plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
# plt.show()
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("SE", bbox_inches='tight')


fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
ax.scatter(S, I, c=ACTIONS, s=60, alpha=1, edgecolors='none')
ax.set_ylabel('S', fontsize=40)
ax.set_xlabel('I', fontsize=40)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_ylim(0,1e5)
ax.set_xlim(0,1e5)
plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
# plt.show()
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("SI", bbox_inches='tight')


fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
ax.scatter(E, I, c=ACTIONS, s=60, alpha=1, edgecolors='none')
ax.set_ylabel('E', fontsize=40)
ax.set_xlabel('I', fontsize=40)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_ylim(0,1e5)
ax.set_xlim(0,1e5)
plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
# plt.show()
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig("EI", bbox_inches='tight')



def plot_fig(STATES, ACTIONS, fig_name):
    S = STATES[:,0]
    E = STATES[:,1]
    I = STATES[:,2]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
    ax.scatter(S, E, c=ACTIONS, s=60, alpha=1, edgecolors='none')
    ax.set_xlabel('S', fontsize=40)
    ax.set_ylabel('E', fontsize=40)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_ylim(0,1e5)
    ax.set_xlim(0,1e5)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
    # plt.show()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(fig_name + '_' + "SE", bbox_inches='tight')


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
    ax.scatter(S, I, c=ACTIONS, s=60, alpha=1, edgecolors='none')
    ax.set_xlabel('S', fontsize=40)
    ax.set_ylabel('I', fontsize=40)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_ylim(0,1e5)
    ax.set_xlim(0,1e5)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
    # plt.show()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(fig_name + '_' + "SI", bbox_inches='tight')


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
    ax.scatter(E, I, c=ACTIONS, s=60, alpha=1, edgecolors='none')
    ax.set_xlabel('E', fontsize=40)
    ax.set_ylabel('I', fontsize=40)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_ylim(0,1e5)
    ax.set_xlim(0,1e5)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
    # plt.show()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(fig_name + '_' + "EI", bbox_inches='tight')


STATES_0, STATES_1, STATES_2 = [], [], []
ACTIONS_0, ACTIONS_1, ACTIONS_2 = [], [], []

for i, a in enumerate(ACTIONS[0]):
    if a==0:
        STATES_0.append(STATES[i].tolist())
        ACTIONS_0.append(a)
    elif a==1:
        STATES_1.append(STATES[i].tolist())
        ACTIONS_1.append(a)
    elif a==2:
        STATES_2.append(STATES[i].tolist())
        ACTIONS_2.append(a)
    else:
        pass
pol_dict = {0:'Lockdown', 1:'SocialDist', 2:'Open'}

plot_fig(np.array(STATES_0), ACTIONS_0, pol_dict[0])
plot_fig(np.array(STATES_1), ACTIONS_1, pol_dict[1])
plot_fig(np.array(STATES_2), ACTIONS_2, pol_dict[2])



fig, ax = plt.subplots(nrows=3, ncols=3, figsize = (20,20))
for i in range(3):
    for j in range(3):
        if j==0:
            data_S, data_A = STATES_0, ACTIONS_0
        elif j==1:
            data_S, data_A = STATES_1, ACTIONS_1
        elif j==2:
            data_S, data_A = STATES_2, ACTIONS_2
        S = np.array(data_S)[:,0]
        E = np.array(data_S)[:,1]
        I = np.array(data_S)[:,2]
        if i==0:
            X, Y = S, E
            label = ['S', 'E']
        elif i==1:
            X, Y = S, I
            label = ['S', 'I']
        elif i==2:
            X, Y = E, I
            label = ['E', 'I']
        ax[i,j].scatter(X, Y, c=data_A, s=60, alpha=1, edgecolors='none')
        ax[i,j].set_xlabel(label[0], fontsize=20)
        ax[i,j].set_ylabel(label[1], fontsize=20)
        ax[i,j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax[i,j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax[i,j].set_ylim(0,1e5)
        ax[i,j].set_xlim(0,1e5)
        ax[i,j].tick_params(axis='both', which='major', labelsize=15)
        ax[i,j].tick_params(axis='both', which='minor', labelsize=15)
        if i==0:
            ax[i,j].set_title(pol_dict[j], fontsize=15)
        # ax[i,j].yticks(fontsize=20)     
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(0.9, 1), loc=1)
# plt.show()
plt.savefig('Overall', bbox_inches='tight')