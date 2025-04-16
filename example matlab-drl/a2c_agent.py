import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
import logging

# ==================================
# === Logging Setup (Add This) ===
# ==================================
log_file = 'a2c_agent_debug.log'
logger = logging.getLogger('a2c_agent') # Use a specific name for your logger
logger.setLevel(logging.DEBUG) # Capture all levels of messages

# Prevent adding handlers multiple times if the script is reloaded
if not logger.handlers:
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    logger.info("--- Logger initialized ---")
# ==================================
# === End Logging Setup ===
# ==================================

# === Create environment and agent ===
env = gym.make("Pendulum-v1")
dummy_env = DummyVecEnv([lambda: env])
model = A2C("MlpPolicy", dummy_env, device="cpu", verbose=0)

# === Create rollout buffer ===
n_steps = 5
buffer = RolloutBuffer(
    buffer_size=n_steps,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=model.device,
    gamma=model.gamma,
    gae_lambda=model.gae_lambda,
)

# === Globals ===
step_count = 0
obs_last = None
action_last = None
value_last = None
log_prob_last = None

# === Environment Reset ===
def reset_env():
    global obs_last
    obs_last, _ = env.reset()
    return obs_last.tolist()  # 👈 Return plain list for MATLAB compatibility


# # === Choose action ===
# def get_action(obs):
#     global action_last, value_last, log_prob_last
#     obs = np.array(obs, dtype=np.float32).reshape((1, -1))
#     obs_tensor = torch.tensor(obs).float().to(model.device)
#     with torch.no_grad():
#         action_tensor, value_tensor, log_prob_tensor = model.policy.forward(obs_tensor)
#     action_last = action_tensor
#     value_last = value_tensor
#     log_prob_last = log_prob_tensor
#     return action_tensor.cpu().numpy()[0]


# === Choose action (Modified Function) ===
def get_action(obs):
    logger.debug("--- get_action called ---")
    global action_last, value_last, log_prob_last
    try:
        # Log the input *before* any processing
        logger.debug(f"Received obs type: {type(obs)}")
        logger.debug(f"Received obs value: {obs}")

        # --- Your original logic ---
        obs_np = np.array(obs, dtype=np.float32).reshape((1, -1))
        logger.debug(f"obs converted to numpy: {obs_np}, Shape: {obs_np.shape}, Dtype: {obs_np.dtype}")

        obs_tensor = torch.tensor(obs_np).float().to(model.device)
        logger.debug(f"obs converted to tensor: {obs_tensor}, Shape: {obs_tensor.shape}, Device: {obs_tensor.device}")

        with torch.no_grad():
            action_tensor, value_tensor, log_prob_tensor = model.policy.forward(obs_tensor)
            logger.debug(f"Policy output -> action: {action_tensor}, value: {value_tensor}, log_prob: {log_prob_tensor}")

        action_last = action_tensor
        value_last = value_tensor
        log_prob_last = log_prob_tensor

        action_to_return = action_tensor.cpu().numpy()[0]
        logger.debug(f"Action to return (numpy): {action_to_return}, Type: {type(action_to_return)}")
        # --- End of original logic ---

        logger.debug("--- get_action finished successfully ---")
        return action_to_return

    except NameError as ne:
        # Specifically log NameErrors, which you encountered
        logger.error(f"NameError occurred in get_action: {ne}", exc_info=True)
        logger.error("Check if the 'obs' variable was correctly passed from MATLAB via the struct.")
        raise # Re-raise the exception so MATLAB knows it failed
    except Exception as e:
        # Log any other unexpected exceptions
        logger.error("An unexpected error occurred in get_action", exc_info=True) # exc_info=True adds traceback
        logger.debug("--- get_action finished with error ---")
        raise # Re-raise the exception

# === Step in environment ===
def my_step(action):
    next_obs, reward, terminated, truncated, _ = env.step(np.array(action))
    done = terminated or truncated
    return next_obs.tolist(), float(reward), bool(done)  # 👈 Return clean native types


# === Store experience ===
def store_transition(reward, done, next_obs):
    global step_count, obs_last
    reward = np.array([reward], dtype=np.float32)
    done = np.array([done], dtype=bool)
    next_obs = np.array(next_obs, dtype=np.float32).reshape((1, -1))
    buffer.add(obs_last.reshape((1, -1)), action_last, reward, done, value_last, log_prob_last)
    step_count += 1
    obs_last = next_obs
    if step_count % n_steps == 0:
        with torch.no_grad():
            last_val = model.policy.predict_values(torch.tensor(next_obs).float().to(model.device))
        buffer.compute_returns_and_advantage(last_val, dones=done)
        model.train()

# === Save/Load Model ===
def save_model(path="a2c_pendulum"):
    model.save(path)

def load_model(path="a2c_pendulum"):
    global model
    model = A2C.load(path, env=dummy_env)
