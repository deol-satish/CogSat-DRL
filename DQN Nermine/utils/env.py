import gymnasium
import numpy as np
import matlab.engine
from gymnasium.spaces import MultiDiscrete, Dict, Box
import logging
import json
import math
from datetime import datetime, timedelta, timezone

env_name = "NermineCogSatEnv-v1"

# Configure the logger
logging.basicConfig(
    filename='state_log.txt',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'  # Overwrites the file each time
)

 
class CogSatEnv(gymnasium.Env):
    """Gymnasium environment for MATLAB-based Cognitive Satellite Simulation"""
 
    def __init__(self, env_config=None, render_mode=None):
        super(CogSatEnv, self).__init__()
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gymnasium.envs.registration.EnvSpec("CogSatEnv-v1")
            
 
        # Start MATLAB engine and set path
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.eng.pwd(), nargout=0)  # ensure working directory is set

        self.eng.addpath(r'./matlab_code', nargout=0)

        # Initialize the MATLAB scenario
        self.eng.eval("initialiseScenario", nargout=0)
        self.eng.eval("resetScenario", nargout=0)


        self.tIndex = 0
        self.timelength = self.eng.eval("length(ts)", nargout=1)
        self.leoNum = int(self.eng.workspace['leoNum'])
        self.geoNum = int(self.eng.workspace['geoNum'])
        self.NumLeoUser = int(self.eng.workspace['NumLeoUser'])
        self.NumGeoUser = int(self.eng.workspace['NumGeoUser'])
        self.curLEO_User_id = 0
        self.reward = -32.4115512468957

        self.LeoChannels = int(self.eng.workspace['numChannels'])
        self.GeoChannels = self.eng.workspace['NumGeoUser']

        self.ChannelListLeo = self.eng.workspace['ChannelListLeo']
        self.ChannelListGeo = self.eng.workspace['ChannelListGeo']

        self.intial_obs = {
            "utc_time": np.array([0], dtype=np.int64),
            "freq_lgs_leo": np.random.uniform(20.0, 22.0, size=(self.NumLeoUser,)).astype(np.float64),
            "freq_ggs_geo": np.random.uniform(20.0, 22.0, size=(self.NumGeoUser,)).astype(np.float64),
        }         
        
 
        # Define action and observation space
        self.action_space = gymnasium.spaces.Discrete(self.LeoChannels)  # Select a channel index for one LEO (for example)


        # Observation space structure
        self.observation_space = Dict({
            "utc_time": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "freq_lgs_leo": Box(low=-np.inf, high=np.inf, shape=(self.NumLeoUser,), dtype=np.float64),
            "freq_ggs_geo": Box(low=-np.inf, high=np.inf, shape=(self.NumGeoUser,), dtype=np.float64),
        })
    
    
    def get_matlab_ts(self):
        """
        Get the MATLAB timestamp as a list of strings.
        """
        ts_str = self.eng.eval("cellstr(datestr(ts, 'yyyy-mm-ddTHH:MM:SS'))", nargout=1)
        python_datetimes = [datetime.fromisoformat(s) for s in ts_str]
        timestamps = [dt.timestamp() for dt in python_datetimes]
        return timestamps
    

    


    def get_state_from_matlab(self):
        # Log cur_state_from_matlab
        logging.info("=== Current State ===")
        # logging.info(json.dumps(cur_state_from_matlab, indent=2))
        """Reset the environment and initialize the buffer."""

        self.ts = self.get_matlab_ts()

        self.FreqAlloc = np.array(self.eng.workspace['FreqAlloc'])
        self.LEOFreqAlloc = self.FreqAlloc[:10,:]
        self.GEOFreqAlloc = self.FreqAlloc[10:20,:]

        cur_obs = self.intial_obs.copy()

        cur_obs["utc_time"] = np.array([self.ts[self.tIndex]], dtype=np.int64)
        cur_obs["freq_lgs_leo"] = np.array(self.LEOFreqAlloc[:,self.tIndex], dtype=np.float64)
        cur_obs["freq_ggs_geo"] = np.array(self.GEOFreqAlloc[:,self.tIndex], dtype=np.float64)

        logging.info("self.tIndex: %s",self.tIndex)

        # Log utc_time
        logging.info("utc_time: %s", cur_obs["utc_time"].tolist())

        # Log freq_lgs_leo
        logging.info("freq_lgs_leo: %s", cur_obs["freq_lgs_leo"].tolist())

        # (Optional) Validate against observation_space
        assert self.observation_space.contains(cur_obs), "cur_obs doesn't match the observation space!"

        return cur_obs
    

 
    def step(self, action):
        """
        Apply action and return (observation, reward, terminated, truncated, info)
        """
        print("*-"*50)
        print("Step Started")
        logging.info("=== Step Started ===")
        # Access the variable from MATLAB workspace
        channel_list_leo = self.eng.workspace['ChannelListLeo']

        # Convert MATLAB array to NumPy array
        channel_list_leo_np = np.array(channel_list_leo)

        Serv_idxLEO = np.array(self.eng.workspace['Serv_idxLEO'])
        

        self.cur_leo_sat_id = int(Serv_idxLEO[self.curLEO_User_id, self.tIndex]) - 1

        self.tIndex = int(self.tIndex)
        

        print("Action taken: ", action)
        logging.info("=== Action Taken === %s", action)

        print("Current LEO User ID: ", self.curLEO_User_id)
        logging.info("=== Current LEO User ID === %s", self.curLEO_User_id)

        print("self.tIndex: ", self.tIndex)
        logging.info("=== Current Time Index === %s", self.tIndex)

        print("Current LEO Satellite ID: ", self.cur_leo_sat_id)
        logging.info("=== Current LEO Satellite ID === %s", self.cur_leo_sat_id)


        channel_list_leo_np[self.curLEO_User_id, self.cur_leo_sat_id, self.tIndex] = int(action)
        self.eng.workspace['ChannelListLeo'] = matlab.double(channel_list_leo_np)

        print("Updated ChannelListLeo: ", np.array(self.eng.workspace['ChannelListLeo'])[self.curLEO_User_id, self.cur_leo_sat_id, self.tIndex])




        self.eng.eval("stepScenario", nargout=0)
        next_observation = self.get_state_from_matlab()     
        
        print("Next Observation: ", next_observation)
        logging.info("=== Next Observation === %s", next_observation)   
         
        terminated = False
        truncated = False

        SINR = np.array(self.eng.workspace['SINR'])
        print("SINR[:,self.tIndex]: ", SINR[:,self.tIndex])

        reward = -32.4115512468957

        reward = np.sum(SINR[:,self.tIndex])
        self.reward = reward
        print("Reward: ", reward)
        logging.info("=== Reward === %s", reward)

        self.curLEO_User_id += 1

        if self.curLEO_User_id >= self.NumLeoUser:
            self.curLEO_User_id = 0
            self.tIndex += 1
            if self.tIndex >= self.timelength:
                terminated = True
                print("Episode finished after {} timesteps".format(self.tIndex))
                logging.info("=== Episode finished after %s timesteps ===", self.tIndex)
                self.eng.eval("P07_Plotting", nargout=0)

        info = {}

        print("*-"*50)

 
        return next_observation, reward, terminated, truncated, info
 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) 
        # Reset the scenario
        self.eng.eval("resetScenario", nargout=0)

        self.ChannelListLeo = self.eng.workspace['ChannelListLeo']
        self.ChannelListGeo = self.eng.workspace['ChannelListGeo']

        self.tIndex = 0
        self.done = 0
        self.curLEO_User_id = 0

        observation = self.get_state_from_matlab()
        print("++++===== ENV RESET+++===")
 
        return observation, {}
 
    def render(self):
        print("Rendering is handled in MATLAB viewer.")
 
    def close(self):
        print("Saving MATLAB Data.")
        logging.info("=== Saving MATLAB Data ===")
        # self.eng.eval("P07_Plotting", nargout=0)
        self.eng.quit()
    