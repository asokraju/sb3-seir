import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class TSeirEnv(gym.Env):
    """
    Description:
            Each city's population is broken down into four compartments --
            Susceptible, Exposed, Infectious, and Removed -- to model the spread of
            COVID-19.
    Source:
            Code modeled after cartpole.py from
            github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    
    Time:
            discretizing_time: time in minutes used to discretizing the model
            sampling_time: time in days that we sample from the system
            sim_length: time in days
            
            
    Observation*:
            Type: Box(4,)
            Num     Observation       Min     Max
            0       Susceptible       0       Total Population
            1       Exposed           0       Total Population
            2       Infected          0       Total Population
            3       Recovered         0       Total Population
            4       Day               0        25
            
    
    Actions*:
            Type: Box(4,), min=0 max=2
            Num     Action                                   Change in model                Crowd density
            0       Lockdown                                 affect transmission rate       0
            1       Social distancing                        affect transmission rate       0.5-1 = 0.75     
            2       No Social distancing                     affect transmission rate       1-5   = 1.5
            
    Reward:
            reward = weight * economic cost + (1-weight) * public health cost
            
            Economic cost:
            Num       Action                                    Crowd density               cost
            0         Lockdown                                  0                           1.0
            1         Social distancing                         0.5-1 = 0.75                0.25
            2         No Social distancing (regular day)        1-5   = 1.5                 0.0
            Health cost:                                min                     max
                1.0 - 0.00001* number of infected      0.0                      1.0
    weight:
        a user defined weight. Default 0.5
    Episode Termination:
            Episode length (time) reaches specified maximum (end time)
            The end of analysis period is ~170 days
    """


    metadata = {'render.modes': ['console']}

    def __init__(
        self, 
        discretizing_time = 5, 
        sampling_time = 7, 
        sim_length = 175, 
        weight = 0.5, 
        theta = 113.92, 
        inital_state =  [99666., 81., 138., 115., 0.], 
        state_normalization = True,
        validation = False,
        noise = False,
        noise_percent = 0):
        super(TSeirEnv, self).__init__()

        self.dt           = discretizing_time/(24*60)
        self.Ts           = sampling_time
        self.time_steps   = int((self.Ts) / self.dt)
       
        self.popu         = 1e5 #100000
        self.state_normalization = state_normalization
        self.trainNoise   = False
        self.weight       = weight #reward weighting
        self.inital_state = np.array(inital_state, dtype=float)
        self.validation   = validation

        #model paramenters
        self.theta    = np.full(shape=1, fill_value=theta, dtype=float)  

        self.d            = np.full(shape=1, fill_value=1/24, dtype=float)

        self.sigma        = 1.0/5   # needds to be changed?

        # Recovery rate
        self.gamma        = 0.05    # needs to be changed?

        # total number of actions
        self.n_actions    = 3

        # Crowd densities
        self.rho          = np.array([0.044, 0.25, 1.], dtype=float)
        self.infection_rate = self.theta * (self.d ** 2) * self.rho
        # Resulting Infection rate =  beta = rho*theta*d^2 = [0.009, 0.049, 0.196]
        # there is some non-zero infection rate duction lockdown --> due to necessary supply chains that we cannot stop for survival
        # Reproduction number =  R0 = beta/gamma = [0.18, 0.98, 3.92]
        # If R0 is less than one the disease will die out, and if R0>1 the disease will increase exponentially
        #Economic costs 
        self.eco_costs    = np.array([1, 0.2, 0.0], dtype=float) 

        #gym action space and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, np.inf, shape=(5,), dtype=np.float64)

        #Total number of simulation days
        self.sim_length   = sim_length
        self.daynum       = 0

        # noise
        self.noise         = noise
        self.noise_percent = noise_percent

        #seeding
        self.seed()

        #memory to save the trajectories
        self.state_trajectory  = []
        self.action_trajectory = []
        self.rewards           = []
        self.weekly_rewards    = []
        self.count             = 0
        
        # initialize state
        self.get_state()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        if not self.validation:
            # init_E = np.random.normal(self.inital_state[1],self.inital_state[1]*0.1) 
            # # np.random.randint(self.inital_state[1]*0.1, high=self.inital_state[1]*2)
            # init_I = np.random.normal(self.inital_state[2],self.inital_state[1]*0.1) 
            # #np.random.randint(self.inital_state[2]*0.1, high=self.inital_state[2]*2)
            # init_R = np.random.normal(self.inital_state[3],self.inital_state[1]*0.1) 
            # #np.random.randint(self.inital_state[3]*0.1, high=self.inital_state[3]*2)
            # init_S = self.popu - init_E - init_I - init_R
            # self.state = np.array([init_S, init_E, init_I, init_R], dtype=float)
            self.state = self.random_uniform_state()
        else:
            self.state = self.inital_state
        self.state_trajectory.append(list(self.state))

    def random_uniform_state(self):
        S = np.random.uniform(low=0.0, high=self.popu)
        E = np.random.uniform(low=0.0, high=self.popu-S)
        I = np.random.uniform(low=0.0, high=self.popu-(S+E))
        R = self.popu-(S+E+I)
        T = 0
        return np.array([S,E,I,R, T])

    def set_state(self, state):
        err_msg = "%s is Invalid. S+E+I+R not equal to %s"  % (state, self.popu)
        assert self.popu==sum(state[:-1]), err_msg
        err_msg = "%s should be of the form (S,E,I,R,T), where T=0" % (state)
        assert state.shape[0]==5, err_msg
        self.state = state
    
    def normalize_state(self,state):
        if self.state_normalization:
            S, E, I, R, T = state[0], state[1], state[2], state[3], state[4]
            S, E, I, R, T = S/self.popu, E/self.popu, I/self.popu, R/self.popu, float(T)/float(self.sim_length)
            return np.array([S, E, I, R, T], dtype=float)
        else:
            return state

    def mini_step(self, rho, day):

        # action should be with in 0 - 2
        # 
        beta = self.theta * (self.d ** 2) * rho
        S, E, I, R, T = self.state

        dS = - (beta) * I * S / self.popu
        dE = - dS - (self.sigma * E)
        dI = (self.sigma * E) - (self.gamma * I)
        dR = (self.gamma * I)

        new_S = S + self.dt * dS
        new_E = E + self.dt * dE
        new_I = I + self.dt * dI
        new_R = R + self.dt * dR

        return np.array([new_S, new_E, new_I, new_R, float(self.daynum)+float(day)], dtype =float)

    def step(self, action):

        for ts in range(self.time_steps):
            self.state = self.mini_step(self.rho[action], day=(ts+1)*self.dt)

            # saving the states and actions in the memory buffers
            self.state_trajectory.append(list(self.state))
            self.action_trajectory.append(action)
            self.count += 1
        self.daynum += self.Ts
        # Costs
        # action represent the crowd density, so decrease in crowd density increases the economic cost
        economicCost = self.eco_costs[action] * self.Ts * 0.91

        # Public health Cost increases with increase in Infected people.
        # publichealthCost   =  (1.45e-5 * (self.state[2]+self.state[3])) * self.Ts
        Delta_S  =  self.state_trajectory[-1-self.time_steps][0] - self.state_trajectory[-1][0]
        publichealthCost = Delta_S/620.0
        
        # Rewards
        reward = - self.weight * economicCost - (1. - self.weight) * publichealthCost
        reward = reward / 150.
        self.weekly_rewards.append(reward)
        # Check if episode is over
        done = bool(self.daynum >= self.sim_length)

        # saving the states and actions in the memory buffers
        #self.state_trajectory.append(list(self.state))
        #self.action_trajectory.append(action)
        for _ in range(self.time_steps):
            self.rewards.append(reward)
        if not self.noise:
            return self.normalize_state(self.state), reward, done, {}
        else:
            S, E, I, R = self.state[0], self.state[1], self.state[2], self.state[3]
            I = (1 - (self.noise_percent / 100) ) * I
            S = (1 + (self.noise_percent / 100) ) * S
            noisy_state = np.array([S, E, I, R, self.state[4]], dtype =float)
            return self.normalize_state(noisy_state), reward, done, {}
        
    def reset(self):

        # reset to initial conditions
        self.daynum = 0

        #memory reset
        self.state_trajectory  = []
        self.action_trajectory = []
        self.rewards           = []
        self.weekly_rewards    = []
        self.count = 0
        self.get_state()

        return self.normalize_state(self.state)
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print("not implemented")

    def close(self):
        pass
