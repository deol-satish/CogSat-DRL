import gymnasium
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete
import numpy as np
import pandas as pd
import random
import logging


from utils.settings import channelFreqs as FREQUENCY
from utils.settings import n_leo, n_geo, n_leo_users, n_geo_users

import math

from utils.df_preprocess import get_df_processed


global df_obs_space 
df_obs_space = pd.Dataframe()

class CogSatDSAEnv(gymnasium.Env):
    # env_config=None, render_mode=None added to match with SB3 configurations
    # to use with Ray RLlib "self, env_config" is enough
    def __init__(self, env_config=None, render_mode=None):
        super(CogSatDSAEnv, self).__init__()
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gymnasium.envs.registration.EnvSpec("CogSatDSAEnv-v1")
        # Actions
        # Channel Freq - 10 frequencies with 0 as None
        self.action_space = MultiDiscrete([11]*4)

        # Observation space
        self.observation_space = Dict({
            "utc_time": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "leo_pos": Box(low=-np.inf, high=np.inf, shape=(n_leo,), dtype=np.float64),
            "leo_rssi": Box(low=-np.inf, high=np.inf, shape=(n_leo*n_leo_users,), dtype=np.float64),
            "leo_sinr": Box(low=-np.inf, high=np.inf, shape=(n_leo*n_leo_users,), dtype=np.float64),
            "geo_rssi": Box(low=-np.inf, high=np.inf, shape=(n_geo*n_geo_users,), dtype=np.float64),            
            "geo_sinr": Box(low=-np.inf, high=np.inf, shape=(n_geo*n_geo_users,), dtype=np.float64),
        })
        

        self.terminated = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.terminated = False

        # Observation space
        # Apply actions for each agent, return observations, rewards, dones, and optional info
        # observation = {
        #     "utc_time": np.array([self.utc_time], dtype=np.int64),
        #     "beam_positions": np.random.uniform(low=-1000, high=1000, size=(28,)),
        #     "previous_actions": np.random.randint(low=0, high=9, size=(14,), dtype=np.int64)
        # }

        df_obs_space = get_df_processed()
        
        observation = {
            "utc_time": np.array([df_obs_space['Time'].iloc[0]], dtype=np.int64),
            "beam_positions": np.random.uniform(low=-1000, high=1000, size=(28,)),
            "previous_actions": np.random.randint(low=0, high=9, size=(14,), dtype=np.int64)
        }


        return observation, {}

    def save_env_state(self):
        state = {
            'utc_time': self.utc_time,
            'leo1_positions': [(turtle.xcor(), turtle.ycor()) for turtle in self.leo1.all_turtles],
            'leo2_positions': [(turtle.xcor(), turtle.ycor()) for turtle in self.leo2.all_turtles],
            'leo3_positions': [(turtle.xcor(), turtle.ycor()) for turtle in self.leo3.all_turtles],
            'leo4_positions': [(turtle.xcor(), turtle.ycor()) for turtle in self.leo4.all_turtles],
            'leo1_directions': [turtle.heading() for turtle in self.leo1.all_turtles],
            'leo2_directions': [turtle.heading() for turtle in self.leo2.all_turtles],
            'leo3_directions': [turtle.heading() for turtle in self.leo3.all_turtles],
            'leo4_directions': [turtle.heading() for turtle in self.leo4.all_turtles],
            'leo1_pen': [turtle.isdown() for turtle in self.leo1.all_turtles],
            'leo2_pen': [turtle.isdown() for turtle in self.leo2.all_turtles],
            'leo3_pen': [turtle.isdown() for turtle in self.leo3.all_turtles],
            'leo4_pen': [turtle.isdown() for turtle in self.leo4.all_turtles],
            'leo1_direction': self.leo1_direction,
            'leo2_direction': self.leo2_direction,
            'leo3_direction': self.leo3_direction,
            'leo4_direction': self.leo4_direction,
            'leo_step': self.leo_step,
            'terminated': self.terminated,
        }
        return state

    def restore_env_state(self, state):
        self.utc_time = state['utc_time']

        for turtle, pos, heading, pen in zip(self.leo1.all_turtles, state['leo1_positions'], state['leo1_directions'],
                                             state['leo1_pen']):
            turtle.penup()
            turtle.goto(pos)
            turtle.setheading(heading)
            if pen:
                turtle.pendown()

        for turtle, pos, heading, pen in zip(self.leo2.all_turtles, state['leo2_positions'], state['leo2_directions'],
                                             state['leo2_pen']):
            turtle.penup()
            turtle.goto(pos)
            turtle.setheading(heading)
            if pen:
                turtle.pendown()

        for turtle, pos, heading, pen in zip(self.leo3.all_turtles, state['leo3_positions'], state['leo3_directions'],
                                             state['leo3_pen']):
            turtle.penup()
            turtle.goto(pos)
            turtle.setheading(heading)
            if pen:
                turtle.pendown()

        for turtle, pos, heading, pen in zip(self.leo4.all_turtles, state['leo4_positions'], state['leo4_directions'],
                                             state['leo4_pen']):
            turtle.penup()
            turtle.goto(pos)
            turtle.setheading(heading)
            if pen:
                turtle.pendown()

        self.leo1_direction = state['leo1_direction']
        self.leo2_direction = state['leo2_direction']
        self.leo3_direction = state['leo3_direction']
        self.leo4_direction = state['leo4_direction']
        self.leo_step = state['leo_step']
        self.terminated = state['terminated']

    def step(self, actions):
        leo_beam_positions = []

        distance = self.leo_speed

        self.leo1.move(distance, angle=(-45+self.leo_step))
        self.leo2.move(distance, angle=(45-self.leo_step))
        if self.utc_time > 225:
            self.leo3.move(distance, angle=(-45+self.leo_step))
            self.leo4.move(distance, angle=(45-self.leo_step))
        else:
            self.leo3.move(distance, angle=-45)
            self.leo4.move(distance, angle=45)

        for leo_beam in range(7):
            leo_beam_positions.append(np.array(self.leo1.all_turtles[leo_beam].xcor()))
            leo_beam_positions.append(np.array(self.leo1.all_turtles[leo_beam].ycor()))
            leo_beam_positions.append(np.array(self.leo2.all_turtles[leo_beam].xcor()))
            leo_beam_positions.append(np.array(self.leo2.all_turtles[leo_beam].ycor()))

        if (self.leo3.all_turtles[0].xcor() > self.screen_edge) or \
                    (self.leo4.all_turtles[0].xcor() > self.screen_edge):
            self.terminated = True

        self.leo_step += 0.005   # define the intensity of angle
        self.utc_time += 1
        # print(f'env time step {self.utc_time}')

        observation = {
        "utc_time": np.array([self.utc_time], dtype=np.int64),
        "beam_positions":np.array(leo_beam_positions, dtype=np.float64),
        "previous_actions": np.array(actions,dtype=np.int64)
        }

        ## Reward calculation
        leo_bandwidth = 0.2  # beam bandwidth in MHz

        # list with leo user capacity
        self.leo_user_capacity = []

        # list with geo user capacity
        self.geo_user_capacity = []

        # list with leo interference to GEO users
        self.leo_to_geo_user_interference = []

        self.leo_user_interference = []

        leo1_per_beam_capacity = {}
        leo2_per_beam_capacity = {}

        # Calculate the SINR of GEO users
        # This is were work is ongoing
        # P_watts = 10 ** (dBW / 10)  # convert dBW to watts

        for user, geo_beam in self.geo_system.geo_user_beam.items():
            leo_interference, geo_sinr = geointeference.geo_user_interference(user=user, beam=geo_beam, \
            l1_turtles=self.leo1.all_turtles, l2_turtles=self.leo2.all_turtles, leo_freq_sub=self.leo1.freq_sub, geo_freq_sub=self.geo_system.freq_sub,
            bandwidth_ratio = len(self.geo_system.freq_sub[geo_beam.fillcolor()]))
            self.leo_to_geo_user_interference.append(leo_interference)
            self.geo_user_capacity.append(leo_bandwidth*len(self.geo_system.freq_sub[geo_beam.fillcolor()]) * math.log2(1 + geo_sinr))

        # Calculate the SINR of LEO-1 users
        for user, leo_beam in self.leo_users.leo1_for_user.items():
            user_leo1_sinr, user_leo1_interference = leointeference.leo_user_interference(user=user, leo_beam=leo_beam, \
            sat_altitude=pathloss.leo_satellite_altitude_A,int_sat_altitude=pathloss.leo_satellite_altitude_B, \
            int_satellite_properties=self.leo_users.l2_all_turtles, geo_beams=self.geo_system.geo_beams, \
            leo_freq_sub=self.leo1.freq_sub, geo_freq_sub=self.geo_system.freq_sub)
            leo1_capacity_cal = leo_bandwidth * math.log2(1 + user_leo1_sinr)
            self.leo_user_capacity.append(leo1_capacity_cal)
            self.leo_user_interference.append(user_leo1_interference)

            if leo_beam not in leo1_per_beam_capacity:
                leo1_per_beam_capacity[leo_beam] = [leo1_capacity_cal]
            else:
                leo1_per_beam_capacity[leo_beam].append(leo1_capacity_cal)

        new_leo1_per_beam_capacity = {i: value for i, (_, value) in enumerate(leo1_per_beam_capacity.items(), start=1)}
        new_leo1_per_beam_capacity = {key: float(sum(values)) / len(values) for key, values in
                                      new_leo1_per_beam_capacity.items()}


        # Calculate the SINR of LEO-2 users
        for user, leo_beam in self.leo_users.leo2_for_user.items():
            user_leo2_sinr, user_leo2_interference = leointeference.leo_user_interference(user=user, leo_beam=leo_beam, \
            sat_altitude=pathloss.leo_satellite_altitude_B,int_sat_altitude=pathloss.leo_satellite_altitude_A,\
            int_satellite_properties=self.leo_users.l1_all_turtles, geo_beams=self.geo_system.geo_beams, \
            leo_freq_sub=self.leo2.freq_sub, geo_freq_sub=self.geo_system.freq_sub)
            leo2_capacity_cal = leo_bandwidth * math.log2(1 + user_leo2_sinr)
            self.leo_user_capacity.append(leo2_capacity_cal)
            self.leo_user_interference.append(user_leo2_interference)

            if leo_beam not in leo2_per_beam_capacity:
                leo2_per_beam_capacity[leo_beam] = [leo2_capacity_cal]
            else:
                leo2_per_beam_capacity[leo_beam].append(leo2_capacity_cal)

        new_leo2_per_beam_capacity = {i: value for i, (_, value) in enumerate(leo2_per_beam_capacity.items(), start=1)}
        new_leo2_per_beam_capacity = {key: float(sum(values)) / len(values) for key, values in
                                      new_leo2_per_beam_capacity.items()}


        # weights for reward function
        # reward = Avg.LEO user capacity - weight* Avg.GEO interference
        weight1 = 1
        weight2 = -1e13
        reward = weight1*sum(self.leo_user_capacity)/(len(self.leo_user_capacity))\
                 + weight2*sum(self.leo_to_geo_user_interference)/len(self.leo_to_geo_user_interference)



        # Assigning actions to LEO beams
        for a, leo_beam in zip(actions[0:7], range(7)):
            self.leo1.all_turtles[leo_beam].color(FREQUENCY[a])

        for a, leo_beam in zip(actions[7:14], range(7)):
            self.leo2.all_turtles[leo_beam].color(FREQUENCY[a])

        # # Assigning actions to LEO beams
        # for a, leo_beam in zip(actions[14:21], range(7)):
        #     self.leo3.all_turtles[leo_beam].color(FREQUENCY[a])
        #
        # for a, leo_beam in zip(actions[21:28], range(7)):
        #     self.leo4.all_turtles[leo_beam].color(FREQUENCY[a])

        info = {'avg_leo_user_capacity': self.leo_user_capacity,
                'avg_geo_user_capacity': sum(self.geo_user_capacity)/(len(self.geo_user_capacity)),
                'leo_to_geo_interference': self.leo_to_geo_user_interference,
                'leo_user_interference': self.leo_user_interference,
                'leo1_per_beam_capacity': new_leo1_per_beam_capacity,
                'leo2_per_beam_capacity': new_leo2_per_beam_capacity
                }

        truncated = False
        return observation, reward, self.terminated, truncated, info

    def render(self):
        # Implement viz
        pass


