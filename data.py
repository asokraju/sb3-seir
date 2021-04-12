# import copy
# import numpy as np
# import concurrent.futures 
# import time


# import os
# from itertools import permutations
# import gym
# import numpy as np
# import matplotlib.pyplot as plt

# from stable_baselines3 import PPO
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.callbacks import BaseCallback

# def GenerateStates(N=int(1e4)):
#     States = []
#     popu=1e5
#     for _ in range(N):
#         S = np.random.uniform(low=0.0, high=popu)
#         E = np.random.uniform(low=0.0, high=popu-S)
#         I = np.random.uniform(low=0.0, high=popu-(S+E))
#         R = popu-(S+E+I)
#         state = [S,E,I,R]
#         perms = permutations([S,E,I,R])
#         for p in perms:
#             States.append(list(p))
#     return States
# States = GenerateStates(N=int(1e4))


# def work(model, w, env_id, states):
#     Actions = []
#     Rewards =[]
#     next_states = []
#     States = []
#     log_dir = "/content/new-folder"
#     l=np.shape(states)[0]
#     for _, s in enumerate(states):
#         env_kwargs = {
#             'validation':True,
#             'inital_state' : s,
#             'weight' : w
#             }
#         print("\r", "{}/{}".format(_,l), end="")
#         eval_env = gym.make(env_id,**env_kwargs)
#         # eval_env = Monitor(eval_env, log_dir)
#         obs = eval_env.reset()
#         a = model.predict(obs, deterministic=True)[0]
#         _,r,_,_ = eval_env.step(a)
#         States.append(s)
#         Actions.append(a)
#         Rewards.append(r)
#         next_states.append(obs)
#     return States, Actions, Rewards, next_states

# def workers(model, States, w, env_id, n_cpu=6):

#     seqences = np.array_split(States, n_cpu)
#     w_list =  [w for _ in range(n_cpu)]
#     env_list = [env_id for _ in range(n_cpu)]
#     models_list = [copy.deepcopy(model) for _ in range(n_cpu)]

#     with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:        
#         future_to_samples = {executor.submit(model, w, env_id, states): states for model, w, env_id, states in zip(models_list, w_list, env_list, seqences)}
#     S, A, R, NS = [], [], [], []
#     for future in concurrent.futures.as_completed(future_to_samples):
#         States, Actions, Rewards, next_states = future.result()
#         S.vstack(States)
#         A.vstack(Actions)
#         R.vstack(Rewards)
#         NS.vstack(next_states)

#     return S, A, R, NS

# if __name__ == '__main__':
#     path = r"C:\Users\kkris\Documents\GitHub\sb3-seir\results\21-04-09-16-55\0.5\BaseLine\best_model.zip"
#     load_model = PPO.load(path)
#     States = GenerateStates(N=int(1e4))
#     log_dir = "/content/new-folder"
#     w=0.5
#     # model_list = [clone_model(model) for _ in range(4)]
#     out = workers(load_model, States, w, env_id = 'gym_seir:seir-v0', n_cpu=8)
#     print(out)