# matlab_rl_trainer.py
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # For potential normalization wrapper
import time
import os

# --- Configuration ---
NUM_LEOS = 3          # Must match MATLAB
NUM_CHANNELS = 10     # Must match MATLAB
# Define the shape of the observation vector coming from MATLAB
# Example: Time(1) + GEO Freq(1) + LEOs(Lat(1)+Lon(1)+PrevFreq(1))*NUM_LEOS = 1 + 1 + 3*3 = 11
# **MUST MATCH THE STATE VECTOR CREATED IN MATLAB**
OBSERVATION_SHAPE = (11,) # Adjust this based on your state vector definition!

# A2C Hyperparameters (tune these)
N_STEPS = 128         # Collect this many steps before updating policy
GAMMA = 0.99          # Discount factor
LEARNING_RATE = 7e-4  # Learning rate
ENT_COEF = 0.0        # Entropy coefficient
VF_COEF = 0.5         # Value function coefficient
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64])) # Network architecture

SAVE_MODEL_PATH = "a2c_satellite_freq_trained.zip"
SAVE_VECNORM_PATH = "a2c_satellite_vecnorm.pkl" # If using VecNormalize

class RLTrainerInterface:
    """
    Manages the A2C agent training process interacting with MATLAB.
    """
    def __init__(self, verbose=1):
        print("Python: Initializing RL Trainer Interface...")
        self.num_leos = NUM_LEOS
        self.num_channels = NUM_CHANNELS
        self.obs_shape = OBSERVATION_SHAPE
        self.verbose = verbose

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([self.num_channels] * self.num_leos)

        # Create a dummy environment primarily for the agent's space requirements
        # We won't actually call env.step() on this dummy env.
        def dummy_env_creator():
             env = gym.Env()
             env.observation_space = self.observation_space
             env.action_space = self.action_space
             return env

        # Using DummyVecEnv as SB3 often expects vectorized environments
        self.vec_env = DummyVecEnv([dummy_env_creator])

        # Optional: Wrap with VecNormalize for observation/reward normalization
        # self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)
        # print("Python: Using VecNormalize wrapper.")

        # Initialize the A2C agent for training
        print("Python: Creating new A2C agent...")
        self.agent = A2C(
            "MlpPolicy",
            self.vec_env,
            n_steps=N_STEPS,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            policy_kwargs=POLICY_KWARGS,
            verbose=verbose,
            tensorboard_log="./a2c_satellite_tensorboard/" # Log training stats
        )

        # Manually manage the rollout buffer data collection
        self.agent.setup_model() # Sets up buffer, etc.

        # Internal state for collecting transitions
        self.last_obs = None
        self.step_count_session = 0 # Steps collected since last training update
        self.total_steps_ever = 0
        self.episode_reward = 0
        self.episode_length = 0

        print(f"Python: Agent requires {self.agent.n_steps} steps per training update.")
        print("Python: RL Trainer Interface Initialized.")

    def get_action(self, current_obs_list):
        """
        Gets actions from the agent based on the current state.
        Uses stochastic prediction during training. Stores observation.
        """
        try:
            # Convert list from MATLAB to numpy array, ensure correct dtype and shape
            current_obs_np = np.array(current_obs_list, dtype=np.float32).reshape(self.vec_env.observation_space.shape)

            # If using VecNormalize, normalize the observation
            # Note: self.vec_env.normalize_obs(current_obs_np) often expects a batched input
            # Manually adding batch dim:
            obs_for_agent = current_obs_np.reshape(1, -1) # Add batch dim
            # if isinstance(self.vec_env, VecNormalize):
            #     obs_for_agent = self.vec_env.normalize_obs(obs_for_agent)
            #     # Store the *normalized* obs if vec_env expects it later
            #     self.last_obs = obs_for_agent # Store potentially normalized obs

            self.last_obs = obs_for_agent # Store obs (potentially normalized if VecNorm used)


            # Get action (stochastic during training)
            # The agent expects a batched observation
            action, _states = self.agent.predict(self.last_obs, deterministic=False)

            # Action is usually returned with batch dimension, flatten if needed
            action = action.flatten()

            # Convert numpy actions to standard Python ints
            action_list = [int(a) for a in action]

            if self.verbose > 1:
                print(f"Python: Step {self.total_steps_ever}, Obs (first 5): {current_obs_np.flatten()[:5]}, Action: {action_list}")

            return action_list

        except Exception as e:
            print(f"Python: Error in get_action: {e}")
            # Return a default action (e.g., random or first channel)
            return np.random.randint(0, self.num_channels, size=self.num_leos).tolist()


    def store_experience_and_train(self, reward, next_obs_list, done):
        """
        Stores the experience tuple in the agent's buffer and triggers training
        if enough steps have been collected.
        """
        try:
            self.step_count_session += 1
            self.total_steps_ever += 1
            self.episode_reward += reward
            self.episode_length += 1

            # Convert next_obs list from MATLAB to numpy array
            next_obs_np = np.array(next_obs_list, dtype=np.float32).reshape(self.vec_env.observation_space.shape)

            # --- Prepare data for buffer ---
            # The buffer expects NumPy arrays, often without the batch dimension handled internally
            # Ensure self.last_obs is the correct shape expected by buffer.add
            # (Usually needs the shape matching observation_space, not necessarily batched)
            current_obs_unbatched = self.last_obs.flatten() # Remove batch dim if added in get_action

            # Retrieve the last action (should have been predicted in get_action)
            # We need the action that *led* to this reward and next_state
            # This implies `get_action` must be called *before* this function for a given step.
            # The agent's predict method gives the action, we stored the obs. Need action too.
            # Let's modify get_action to store the action it returned.

            # *** Re-predict action here based on stored self.last_obs? No, that's wrong. ***
            # *** Need to receive the action back from MATLAB or store it in get_action ***
            # Let's modify get_action to store self.last_action

            if not hasattr(self, 'last_action'):
                 print("Python: Error - last_action not stored from get_action call.")
                 # Handle error - maybe skip this step?
                 return

            # If using VecNormalize, normalize observations and potentially reward
            reward_to_store = float(reward)
            next_obs_unbatched = next_obs_np # Start with unnormalized
            # if isinstance(self.vec_env, VecNormalize):
            #     # Normalize next_obs (add/remove batch dim as needed by normalize_obs)
            #     next_obs_batched = next_obs_np.reshape(1, -1)
            #     normalized_next_obs_batched = self.vec_env.normalize_obs(next_obs_batched)
            #     next_obs_unbatched = normalized_next_obs_batched.flatten() # Store unbatched normalized obs
            #
            #     # Normalize reward (expects NumPy array)
            #     reward_to_store = self.vec_env.normalize_reward(np.array([reward]))[0]


            # Add to the agent's rollout buffer
            # Need: obs, action, reward, dones, values, log_probs from policy
            # We only have obs, action, reward, dones directly.
            # The agent calculates values and log_probs internally when needed or during collection.
            # We may need to use buffer.add() carefully or trigger agent's internal collection logic.

            # --- Direct Buffer Interaction (Requires understanding SB3 internals) ---
            # This part is complex and might change between SB3 versions.
            # It needs the observation, action, reward, episode_start flag (done), value estimate, log_prob.
            # Getting value estimate and log_prob requires evaluating the policy.
            policy_outputs = self.agent.policy.predict_values(self.last_obs) # Check method name/signature
            values = policy_outputs # Assuming predict_values gives the value estimate
            # Getting log_prob requires evaluating the action probability
            # log_probs = self.agent.policy.evaluate_actions(self.last_obs, self.last_action) # Check method signature
            # We might not have log_probs easily here without re-evaluating.

            # --- Simplified approach: Rely on agent.collect_rollouts structure ---
            # This might be easier but less standard when MATLAB drives.
            # Let's try manually adding essential parts to the buffer if possible.
            # The RolloutBuffer `add` method signature is roughly:
            # add(obs, action, reward, episode_start, value, log_prob)

            # We might need to predict value and log_prob here, which is inefficient.
            # Re-evaluate action probabilities to get log_prob
            # clipped_actions = self.last_action # Assuming action is already clipped/valid
            # Need observations in the correct format for the policy call
            # _, log_prob, _ = self.agent.policy.evaluate_actions(self.last_obs, clipped_actions) # Check signature

            # Add experience to buffer (simplified - might miss log_prob/value initially)
            # SB3 A2C buffer might populate value/log_prob later during compute_returns_and_advantage
            # Let's assume RolloutBuffer handles this internally or we trigger compute later.
            # We need `episode_start` which is equivalent to `done` flag here.
            self.agent.rollout_buffer.add(
                obs=current_obs_unbatched, # Observation that led to action
                action=self.last_action,   # Action taken
                reward=reward_to_store,    # Received reward
                episode_start=bool(done),  # Whether this step is the start of a new episode (or end of last)
                value=values.flatten(),          # Value estimate for obs (predicted above) - needs check
                log_prob=np.zeros(1)       # Placeholder - log_prob might be complex to get here
                                           # Or obtained via policy.evaluate_actions if feasible
            )


            if self.verbose > 1:
                 print(f"Python: Step {self.total_steps_ever}, Stored Exp: R={reward:.2f}, Done={done}")


            # Check if it's time to train
            if self.step_count_session >= self.agent.n_steps:
                print(f"\nPython: Collected {self.step_count_session} steps. Triggering training...")
                start_train_time = time.time()

                # Compute advantages and returns in the buffer before training
                # Need the value estimate for the *last next_obs*
                with np.DisableगूगलProcessing(): # Use torch.no_grad() for PyTorch
                    last_values = self.agent.policy.predict_values(next_obs_unbatched.reshape(1, -1)) # Check method

                self.agent.rollout_buffer.compute_returns_and_advantage(last_values=last_values.flatten(), dones=np.array([bool(done)]))

                # Perform the training update
                self.agent.train()

                # Reset the buffer and session step count
                self.agent.rollout_buffer.reset()
                self.step_count_session = 0
                end_train_time = time.time()
                print(f"Python: Training finished in {end_train_time - start_train_time:.2f} seconds.\n")


            # Handle episode termination for logging
            if done:
                print(f"Python: Episode finished. Total Steps: {self.total_steps_ever}, Episode Length: {self.episode_length}, Episode Reward: {self.episode_reward:.2f}")
                # Log episode stats (optional, SB3 logger might handle this if used correctly)
                # summary = self.agent.logger.record("rollout/ep_rew_mean", self.episode_reward) # Example logging
                # self.agent.logger.dump(step=self.total_steps_ever)

                # Reset episode trackers
                self.episode_reward = 0
                self.episode_length = 0
                # The buffer handles resets internally based on 'done' flags typically
                # Reset self.last_obs? Depends on how MATLAB handles simulation end/reset.
                # If MATLAB continues, next call to get_action will provide the new obs.

        except Exception as e:
            print(f"Python: Error in store_experience_and_train: {e}")
            import traceback
            traceback.print_exc()

    # --- Modify get_action to store the action ---
    def get_action(self, current_obs_list):
        """
        Gets actions from the agent, stores obs and action internally.
        """
        try:
            current_obs_np = np.array(current_obs_list, dtype=np.float32).reshape(self.vec_env.observation_space.shape)
            obs_for_agent = current_obs_np.reshape(1, -1) # Add batch dim

            self.last_obs = obs_for_agent # Store obs

            # Get action (stochastic)
            action, _states = self.agent.predict(self.last_obs, deterministic=False)

            # Store the action that was chosen
            self.last_action = action # Store the action numpy array (potentially batched)

            action_list = [int(a) for a in action.flatten()]

            if self.verbose > 1:
                print(f"Python: Step {self.total_steps_ever}, Obs (first 5): {current_obs_np.flatten()[:5]}, Action: {action_list}")

            return action_list

        except Exception as e:
            print(f"Python: Error in get_action: {e}")
            # Return default, set last_action to something?
            default_action = np.random.randint(0, self.num_channels, size=(1, self.num_leos))
            self.last_action = default_action
            return default_action.flatten().tolist()
    # --- End modification ---


    def save_model(self):
        """Saves the trained agent."""
        try:
            print(f"Python: Saving trained model to {SAVE_MODEL_PATH}...")
            self.agent.save(SAVE_MODEL_PATH)
            print("Python: Model saved.")
            # If using VecNormalize, save its statistics
            # if isinstance(self.vec_env, VecNormalize):
            #     print(f"Python: Saving VecNormalize stats to {SAVE_VECNORM_PATH}...")
            #     self.vec_env.save(SAVE_VECNORM_PATH)
            #     print("Python: VecNormalize stats saved.")
        except Exception as e:
            print(f"Python: Error saving model: {e}")

# --- Global variable to hold the trainer instance ---
rl_trainer_instance = None

def initialize_trainer(verbose_level=1):
    """Initializes the RL Trainer Interface class."""
    global rl_trainer_instance
    if rl_trainer_instance is None:
        try:
             rl_trainer_instance = RLTrainerInterface(verbose=verbose_level)
             print("Python: Trainer instance created.")
             return True # Indicate success
        except Exception as e:
             print(f"Python: Error creating trainer instance: {e}")
             return False # Indicate failure
    else:
        print("Python: Trainer instance already exists.")
        return True # Already initialized

def get_action_from_trainer(current_obs_list):
    """Gets action by calling the trainer instance's method."""
    global rl_trainer_instance
    if rl_trainer_instance:
        return rl_trainer_instance.get_action(current_obs_list)
    else:
        print("Python: Trainer not initialized. Cannot get action.")
        # Return default random action if not initialized
        return np.random.randint(0, NUM_CHANNELS, size=NUM_LEOS).tolist()


def store_and_train_step(action_list, reward, next_obs_list, done):
     """Stores experience and potentially trains by calling the trainer instance's method."""
     global rl_trainer_instance
     if rl_trainer_instance:
         # Need the action numpy array that was stored internally by get_action
         # MATLAB doesn't need to pass the action back if we store it in Python
         rl_trainer_instance.store_experience_and_train(reward, next_obs_list, done)
     else:
         print("Python: Trainer not initialized. Cannot store experience.")


def save_trained_model():
    """Saves the model by calling the trainer instance's method."""
    global rl_trainer_instance
    if rl_trainer_instance:
        rl_trainer_instance.save_model()
    else:
        print("Python: Trainer not initialized. Cannot save model.")