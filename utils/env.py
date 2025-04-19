import gymnasium as gym
import numpy as np
import matlab.engine
 
class CogSatEnv(gym.Env):
    """Gymnasium environment for MATLAB-based Cognitive Satellite Simulation"""
 
    def __init__(self):
        super(CogSatEnv, self).__init__()
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gym.envs.registration.EnvSpec("CogSatEnv")
 
        # Start MATLAB engine and set path
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.eng.pwd(), nargout=0)  # ensure working directory is set
 
        # Initialize the MATLAB scenario
        self._init_matlab_env()
 
        # Define action and observation space (example setup)
        self.action_space = gym.spaces.Discrete(11)  # Select a channel index for one LEO (for example)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
 
    def _init_matlab_env(self):
        self.sc, self.gsList, self.geoSats, self.geoTx, self.geoTxGimbals, \
        self.leoSats, self.leoTx, self.params = self.eng.initializeScenario(nargout=8)
 
    def step(self, action):
        """
        Apply action and return (observation, reward, terminated, truncated, info)
        """
 
        # Action: e.g., select a new frequency for one LEO
        # For this example, we let stepScenario handle the frequency selection randomly
        current_freqs, reward, state = self.eng.stepScenario(self.leoTx, self.params["channelFreqs"])
 
        # Observation: simplified; in practice, you may compute SINR, access status, etc.
        observation = np.random.random(6).astype(np.float32)
 
        # Reward: simplified; in practice, reward could be SINR or channel access success
        reward = np.random.rand()
 
        terminated = False
        truncated = False
        info = {"frequencies": current_freqs}
 
        return observation, reward, terminated, truncated, info
 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
 
        # Reset the scenario
        self.sc, self.gsList, self.geoSats, self.geoTx, self.geoTxGimbals, \
        self.leoSats, self.leoTx, self.params = self.eng.resetScenario(nargout=8)
 
        # Observation after reset
        observation = np.zeros(6, dtype=np.float32)
 
        return observation, {}
 
    def render(self):
        print("Rendering is handled in MATLAB viewer.")
 
    def close(self):
        self.eng.quit()