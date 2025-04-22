import gymnasium
import numpy as np
import matlab.engine
from gymnasium.spaces import MultiDiscrete, Dict, Box
import logging
import json
import math

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

        # Initialize the MATLAB scenario
        self.eng.eval("initialiseScenario", nargout=0)
        self.eng.eval("resetScenario", nargout=0)

        self.leoNum = int(self.eng.workspace['leoNum'])
        self.geoNum = int(self.eng.workspace['geoNum'])
        self.cities = self.eng.workspace['cities_py']

        self.currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
        self.channelFreqs = self.eng.workspace['channelFreqs']

        self.nuser = len(self.cities)

        self.intial_obs = {
            "utc_time": np.array([0], dtype=np.int64),
            "leo_pos": np.random.randn(self.leoNum * 2).astype(np.float64),  # e.g., [x1, y1, x2, y2, x3, y3]
            "geo_freq": np.random.uniform(10.5, 12.0, size=(self.geoNum,)).astype(np.float64),
            "leo_freq": np.random.uniform(20.0, 22.0, size=(self.leoNum,)).astype(np.float64),
            "leo_access": np.random.randint(0, 2, size=(self.leoNum * self.nuser,)).astype(np.float64),
        }         
        
 
        # Define action and observation space (example setup)
        self.action_space = gymnasium.spaces.Discrete(10)  # Select a channel index for one LEO (for example)
        # Observation space structure
        self.observation_space = Dict({
            "utc_time": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "leo_pos": Box(low=-np.inf, high=np.inf, shape=(self.leoNum *2,), dtype=np.float64),
            "geo_freq": Box(low=-np.inf, high=np.inf, shape=(self.geoNum,), dtype=np.float64),
            "leo_freq": Box(low=-np.inf, high=np.inf, shape=(self.leoNum,), dtype=np.float64),
            "leo_access": Box(low=0, high=1, shape=(self.leoNum* self.nuser,), dtype=np.float64),
        })

    def convert_matlab_state_to_py_state(self, cur_state_from_matlab=None):
        # Log cur_state_from_matlab
        logging.info("=== Current State ===")
        logging.info(json.dumps(cur_state_from_matlab, indent=2))
        """Reset the environment and initialize the buffer."""
        cur_obs = self.intial_obs.copy()
        global step_count
        step_count = 0
        from datetime import datetime, timezone

        # 1. utc_time (convert string to UNIX timestamp in seconds)
        dt = datetime.strptime(cur_state_from_matlab["time"], "%d-%b-%Y %H:%M:%S")
        utc_timestamp = int(dt.timestamp())
        cur_obs["utc_time"] = np.array([utc_timestamp], dtype=np.int64)

        # 2. leo_pos (interleaved lat/lon)
        leo_pos = []
        for i in range(1, self.leoNum + 1):
            leo = cur_state_from_matlab[f"LEO_{i}"]
            leo_pos.extend([leo["Latitude"], leo["Longitude"]])
        cur_obs["leo_pos"] = np.array(leo_pos, dtype=np.float64)

        # 3. geo_freq
        cur_obs["geo_freq"] = np.array([cur_state_from_matlab["GeobaseFreq"]], dtype=np.float64)

        # 4. leo_freq (not in cur_state_from_matlab — fill with zeros or placeholder)
        cur_obs["leo_freq"] = np.zeros(self.leoNum, dtype=np.float64)

        # 5. leo_access (flattened [LEO1_Melb, LEO1_Syd, LEO2_Melb, ..., LEO3_Syd])
        leo_access = []
        for i in range(1, self.leoNum + 1):
            access = cur_state_from_matlab[f"LEO_{i}"]["AccessStatus"]
            leo_access.extend([
                float(access["Melbourne"]),
                float(access["Sydney"])
            ])
        cur_obs["leo_access"] = np.array(leo_access, dtype=np.float64)

        # (Optional) Validate against observation_space
        assert self.observation_space.contains(cur_obs), "cur_obs doesn't match the observation space!"

        return cur_obs
 
    def step(self, action):
        """
        Apply action and return (observation, reward, terminated, truncated, info)
        """

        self.currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
        self.channelFreqs = self.eng.workspace['channelFreqs']

        action = int(action)


        if type(self.currentLEOFreqs) == type(float(0)):
            print("Action taken: ", action)
            logging.info("=== Action Taken === %s", action)
            logging.info("=== currentLEOFreqs === %s",self.currentLEOFreqs)
            logging.info("=== currentLEOFreqs === %s",self.currentLEOFreqs)
            self.eng.workspace['currentLEOFreqs'] = self.channelFreqs[0][action]
            
            


        self.eng.eval("stepScenario", nargout=0)
        
        state = self.eng.workspace['snd_state']

        # Reset the observation
        next_observation = self.convert_matlab_state_to_py_state(state)
        terminated = self.eng.workspace['done']
        reward_matlab = self.eng.workspace['reward']

        reward = 0.0

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
 
        terminated = False
        truncated = False
        info = {"frequencies": np.array(self.currentLEOFreqs)}

        if terminated:
            print("Episode terminated.")
            logging.info("=== Episode Terminated ===")
            self.eng.eval("SaveData", nargout=0)
 
        return next_observation, reward, terminated, truncated, info
 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
 
        # Reset the scenario
        self.eng.eval("resetScenario", nargout=0)

        self.leoNum = int(self.eng.workspace['leoNum'])
        self.geoNum = int(self.eng.workspace['geoNum'])
        self.cities = self.eng.workspace['cities_py']

        self.currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
        self.channelFreqs = self.eng.workspace['channelFreqs']

        self.nuser = len(self.cities)
        done = self.eng.workspace['done']
        state = self.eng.workspace['snd_state']
        # reward = self.eng.workspace['reward']

        # Reset the observation

        observation = self.convert_matlab_state_to_py_state(state)
 
        return observation, {}
 
    def render(self):
        print("Rendering is handled in MATLAB viewer.")
 
    def close(self):
        self.eng.quit()