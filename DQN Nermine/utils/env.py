import gymnasium
import numpy as np
import matlab.engine
from gymnasium.spaces import MultiDiscrete, Dict, Box
import logging
import json
import math
from datetime import datetime, timedelta, timezone

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


        self.timestep = 0
        self.timelength = self.eng.eval("length(ts)", nargout=1)
        self.leoNum = int(self.eng.workspace['leoNum'])
        self.geoNum = int(self.eng.workspace['geoNum'])
        self.NumLeoUser = int(self.eng.workspace['NumLeoUser'])

        self.LeoChannels = self.eng.workspace['numChannels']
        self.GeoChannels = self.eng.workspace['NumGeoUser']

        self.ChannelListLeo = self.eng.workspace['ChannelListLeo']
        self.ChannelListGeo = self.eng.workspace['ChannelListGeo']

        self.intial_obs = {
            "utc_time": np.array([0], dtype=np.int64),
            "freq_lgs_leo": np.random.uniform(20.0, 22.0, size=(self.NumLeoUser,)).astype(np.float64),
        }         
        
 
        # Define action and observation space (example setup)
        # self.action_space = gymnasium.spaces.Discrete(10)  # Select a channel index for one LEO (for example)

        self.action_space = gymnasium.spaces.Box(low=1.0, high=25.0, shape=(self.NumLeoUser,), dtype=np.float32)


        # Observation space structure
        self.observation_space = Dict({
            "utc_time": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "freq_lgs_leo": Box(low=-np.inf, high=np.inf, shape=(self.NumLeoUser,), dtype=np.float64),
        })

    def get_matlab_datetime(self):
        """
        Get the MATLAB timestamp as a list of strings.
        """
        ts_str = self.eng.eval("cellstr(datestr(ts, 'yyyy-mm-ddTHH:MM:SS'))", nargout=1)
        python_datetimes = [datetime.fromisoformat(s) for s in ts_str]
        return python_datetimes
    
    
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

        cur_obs = self.intial_obs.copy()

        cur_obs["utc_time"] = np.array([self.ts[self.timestep]], dtype=np.int64)
        cur_obs["freq_lgs_leo"] = np.array(self.LEOFreqAlloc[:,self.timestep], dtype=np.float64)

        logging.info("self.timestep: %s",self.timestep)

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

         
        terminated = False
        truncated = False

        self.currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
        self.channelFreqs = self.eng.workspace['channelFreqs']

        action = int(action)

        self.leoIndex = self.eng.workspace['leoIndex']


        if type(self.currentLEOFreqs) == type(float(0)):
            print("Action taken: ", action)
            logging.info("=== Action Taken === %s", action)
            logging.info("=== currentLEOFreqs === %s",self.currentLEOFreqs)
            logging.info("=== currentLEOFreqs === %s",self.currentLEOFreqs)
            self.eng.workspace['currentLEOFreqs'] = self.channelFreqs[0][action]
        else:
            currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
            #Modify the values (example change)
            new_values = np.array(currentLEOFreqs).flatten().tolist()
            print("Old Frequencies: ", new_values)

            new_values[int((self.leoIndex -1))] = self.channelFreqs[0][action]

            print("New Frequencies: ", new_values)

            #Update the workspace variable
            self.eng.workspace['currentLEOFreqs'] = matlab.double(new_values)
            
            


        self.eng.eval("stepScenario", nargout=0)

        print("Step Scenario",self.eng.workspace['tIdx'])
        
        state = self.eng.workspace['snd_state']

        # Reset the observation
        next_observation = self.convert_matlab_state_to_py_state(state)
        terminated = self.eng.workspace['done']
        reward_matlab = self.eng.workspace['reward']

        reward = -32.4115512468957

        calc_reward = 0.0

        state['LEO_1']['AccessStatus']['Melbourne']
        if (state['LEO_1']['AccessStatus']['Melbourne'] and state['LEO_1']['AccessStatus']['Sydney']):
            reward = reward_matlab["LEO_1"]['reward']['Melbourne']['snr'] + reward_matlab["LEO_1"]['reward']['Sydney']['snr']
            reward = reward /2
        elif (state['LEO_1']['AccessStatus']['Melbourne'] and not state['LEO_1']['AccessStatus']['Sydney']):
            reward = reward_matlab["LEO_1"]['reward']['Melbourne']['snr']
        elif (not state['LEO_1']['AccessStatus']['Melbourne'] and state['LEO_1']['AccessStatus']['Sydney']):
            reward = reward_matlab["LEO_1"]['reward']['Sydney']['snr']

        # reward = math.log(reward)

        print("Reward: ", reward)
        logging.info("=== Reward === %s", reward)

        
 
        # Action: e.g., select a new frequency for one LEO
        # For this example, we let stepScenario handle the frequency selection randomly
        
        # Observation: simplified; in practice, you may compute SINR, access status, etc.
        
 
        # Reward: simplified; in practice, reward could be SINR or channel access success

        info = {"frequencies": np.array(self.currentLEOFreqs)}

        if terminated or self.eng.workspace['tIdx'] > 179:
            print("Episode terminated.")
            logging.info("=== Episode Terminated ===")
            self.eng.eval("SaveData", nargout=0)

        if self.eng.workspace['tIdx'] % 100 == 0:
            print("Saving Data every 100 steps")
            logging.info("=== Saving Data every 5 epochs ===")
            self.eng.eval("SaveData", nargout=0)
            truncated = True
 
        return next_observation, reward, terminated, truncated, info
 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
 
        # Reset the scenario
        self.eng.eval("resetScenario", nargout=0)

        self.ChannelListLeo = self.eng.workspace['ChannelListLeo']
        self.ChannelListGeo = self.eng.workspace['ChannelListGeo']

        self.timestep = 0
        self.done = 0

        observation = self.get_state_from_matlab()
        print("++++===== ENV RESET+++===")
 
        return observation, {}
 
    def render(self):
        print("Rendering is handled in MATLAB viewer.")
 
    def close(self):
        print("Saving MATLAB Data.")
        logging.info("=== Saving MATLAB Data ===")
        self.eng.eval("SaveData", nargout=0)
        self.eng.quit()
    