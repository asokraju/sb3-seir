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

model = PPO.load("best_model.zip")
def work(path, s):
    load_model = PPO.load(path)
    tot_pop = 1e5
    states = []
    Actions = []
    Rewards =[]
    next_states = []
    E = np.arange(0.0, tot_pop-s, 1000)
    if not E.shape[0]>0:
        E=np.arange(0.0, 1., 1000.)
    for e in E:
        I = np.arange(0.0, tot_pop-s-e, 1000)
        if not I.shape[0]>0:
            I=np.arange(0.0, 1., 1000.)
        for i in I:
            states.append([s,e,i,tot_pop-s-e-i])
            env_kwargs = {'validation':True}
            env_kwargs['inital_state'] = [s,e,i,tot_pop-s-e-i]
            eval_env = gym.make('gym_seir:seir-v0',**env_kwargs)
            obs = eval_env.reset()
            a = load_model.predict(obs, deterministic=True)[0]
            obs,r, _,_ = eval_env.step(a)
            Actions.append(a)
            Rewards.append(r)
            next_states.append(obs)
    return states, Actions, Rewards, next_states

def workers(data):
    # seqences = data.reshape(num_of_seq, -1)
    # list_models = [copy.deepcopy(model) for _ in range(data.shape[0])]
    list_path = ["best_model.zip" for _ in range(data.shape[0])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:        
        t0 = time.perf_counter()
        future_to_samples = {executor.submit(work, model_path, seq): seq for model_path, seq in zip(list_path, data.tolist())}
    S, A, R, NS = [], [], [], []
    for future in concurrent.futures.as_completed(future_to_samples):
        states, Actions, Rewards, next_states = future.result()
        S.append(states)
        A.append(Actions)
        R.append(Rewards)
        NS.append(next_states)
    t1 = time.perf_counter()
    print(t1-t0)
    return S, A, R, NS, t1-t0

if __name__ == '__main__':
    S = np.arange(0.0, 1e5, 1000.)
    # model_list = [clone_model(model) for _ in range(4)]
    S, A, R, NS, t = workers(S)
    data = {
        'states':S,
        'actions':A,
        'rewards':R,
        'next_state':NS,
        'time':t
    }
    savemat('data.mat', data)
