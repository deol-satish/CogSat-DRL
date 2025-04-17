import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import json
# Configure the logger
logging.basicConfig(
    filename='state_log.txt',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'  # Overwrites the file each time
)

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.settings import n_leo,n_leo_users, n_geo, n_geo_users, initial_state

from utils.env import CogSatDSAEnv
env = CogSatDSAEnv()
obs, _ = env.reset()
print("Initial observation:", obs)

action = env.action_space.sample()
obs, reward, done, truncated, _ = env.step(action)
print("Step observation:", obs)





dummy_env = DummyVecEnv([lambda: env])
# Extract the shape of the observation space, which is a Dict
obs_shape = {key: space.shape for key, space in env.observation_space.spaces.items()}
model = A2C("MultiInputPolicy", dummy_env, verbose=0, device="cpu")

n_steps = 5


import torch
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import DictRolloutBuffer

# === Globals ===
step_count = 0
obs_last = None
action_last = None
value_last = None
log_prob_last = None


intial_obs = {
    "utc_time": np.array([0], dtype=np.int64),
    "leo_pos": np.random.randn(n_leo * 2).astype(np.float64),  # e.g., [x1, y1, x2, y2, x3, y3]
    "geo_freq": np.random.uniform(10.5, 12.0, size=(n_geo,)).astype(np.float64),
    "leo_freq": np.random.uniform(20.0, 22.0, size=(n_leo,)).astype(np.float64),
    "leo_access": np.random.randint(0, 2, size=(n_leo * n_leo_users,)).astype(np.float64),
} 


# You need to define these before use
# env = YourCustomEnv()
# dummy_env = YourCustomEnv()
# model = A2C("MultiInputPolicy", dummy_env, ...)
# n_steps = 5  # or any number you choose

buffer = DictRolloutBuffer(
    buffer_size=n_steps,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=model.device,
    gamma=model.gamma,
    gae_lambda=model.gae_lambda
)

def reset_env(initial_state=None):
    # Log initial_state
    logging.info("=== Initial State ===")
    logging.info(json.dumps(initial_state, indent=2))
    """Reset the environment and initialize the buffer."""
    global obs_last
    global step_count
    step_count = 0
    obs_last = dummy_env.reset()
    from datetime import datetime, timezone

    # 1. utc_time (convert string to UNIX timestamp in seconds)
    dt = datetime.strptime(initial_state["time"], "%d-%b-%Y %H:%M:%S")
    utc_timestamp = int(dt.timestamp())
    obs_last["utc_time"] = np.array([utc_timestamp], dtype=np.int64)

    # 2. leo_pos (interleaved lat/lon)
    leo_pos = []
    for i in range(1, n_leo + 1):
        leo = initial_state[f"LEO_{i}"]
        leo_pos.extend([leo["Latitude"], leo["Longitude"]])
    obs_last["leo_pos"] = np.array(leo_pos, dtype=np.float64)

    # 3. geo_freq
    obs_last["geo_freq"] = np.array([initial_state["GeobaseFreq"]], dtype=np.float64)

    # 4. leo_freq (not in initial_state — fill with zeros or placeholder)
    obs_last["leo_freq"] = np.zeros(n_leo, dtype=np.float64)

    # 5. leo_access (flattened [LEO1_Melb, LEO1_Syd, LEO2_Melb, ..., LEO3_Syd])
    leo_access = []
    for i in range(1, n_leo + 1):
        access = initial_state[f"LEO_{i}"]["AccessStatus"]
        leo_access.extend([
            float(access["Melbourne"]),
            float(access["Sydney"])
        ])
    obs_last["leo_access"] = np.array(leo_access, dtype=np.float64)

    # (Optional) Validate against observation_space
    assert env.observation_space.contains(obs_last), "obs_last doesn't match the observation space!"

    return obs_last


def set_state(initial_state=None):
    # Log initial_state
    logging.info("=== Initial State ===")
    logging.info(json.dumps(initial_state, indent=2))
    """Reset the environment and initialize the buffer."""
    global obs_last
    global step_count
    step_count = 0
    from datetime import datetime, timezone

    # 1. utc_time (convert string to UNIX timestamp in seconds)
    dt = datetime.strptime(initial_state["time"], "%d-%b-%Y %H:%M:%S")
    utc_timestamp = int(dt.timestamp())
    obs_last["utc_time"] = np.array([utc_timestamp], dtype=np.int64)

    # 2. leo_pos (interleaved lat/lon)
    leo_pos = []
    for i in range(1, n_leo + 1):
        leo = initial_state[f"LEO_{i}"]
        leo_pos.extend([leo["Latitude"], leo["Longitude"]])
    obs_last["leo_pos"] = np.array(leo_pos, dtype=np.float64)

    # 3. geo_freq
    obs_last["geo_freq"] = np.array([initial_state["GeobaseFreq"]], dtype=np.float64)

    # 4. leo_freq (not in initial_state — fill with zeros or placeholder)
    obs_last["leo_freq"] = np.zeros(n_leo, dtype=np.float64)

    # 5. leo_access (flattened [LEO1_Melb, LEO1_Syd, LEO2_Melb, ..., LEO3_Syd])
    leo_access = []
    for i in range(1, n_leo + 1):
        access = initial_state[f"LEO_{i}"]["AccessStatus"]
        leo_access.extend([
            float(access["Melbourne"]),
            float(access["Sydney"])
        ])
    obs_last["leo_access"] = np.array(leo_access, dtype=np.float64)

    # (Optional) Validate against observation_space
    assert env.observation_space.contains(obs_last), "obs_last doesn't match the observation space!"

    return obs_last


def preprocess_obs(obs):
    """Convert dict observation to torch tensor dict with proper dimensions."""
    tensor_dict = {}
    for key, value in obs.items():
        # Convert numpy array to tensor and ensure it's float32
        tensor_value = torch.as_tensor(value).float().to(model.device)
        # Add batch dimension if needed (SB3 expects batch dimension)
        if len(tensor_value.shape) == 1:
            tensor_value = tensor_value.unsqueeze(0)
        tensor_dict[key] = tensor_value
    return tensor_dict

def get_action(obs):
    global obs_last, action_last, value_last, log_prob_last
    obs_last = obs

    with torch.no_grad():
        obs_tensor = preprocess_obs(obs)
        # SB3 expects flattened observations for MultiInputPolicy
        flattened_obs = model.policy.obs_to_tensor(obs_tensor)[0]
        action_tensor, value_tensor, log_prob_tensor = model.policy.forward(flattened_obs)

    action_last = action_tensor
    value_last = value_tensor
    log_prob_last = log_prob_tensor

    return action_tensor.cpu().numpy().squeeze()  # Remove batch dimension for env


def my_step(action):
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    return next_obs, reward, done


def compute_a2c_loss(policy, rollout_data, value_coef=0.5, entropy_coef=0.01):
    observations = rollout_data.observations
    actions = rollout_data.actions
    returns = rollout_data.returns
    advantages = rollout_data.advantages
    old_log_probs = rollout_data.old_log_prob

    # Get action distribution and value predictions
    dist = policy.get_distribution(observations)
    value_preds = policy.predict_values(observations)

    # Log probs and entropy from the current policy
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # Actor loss
    policy_loss = -(advantages * new_log_probs).mean()

    # Critic loss
    value_loss = torch.nn.functional.mse_loss(returns, value_preds)

    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total_loss


def store_transition(reward, done, next_obs):
    global step_count, obs_last, action_last, value_last, log_prob_last

    reward = np.array([reward], dtype=np.float32)
    done = np.array([done], dtype=bool)

    buffer.add(
        obs_last,
        action_last,
        reward,
        done,
        value_last,
        log_prob_last
    )

    step_count += 1
    obs_last = next_obs

    print("store_transition,, step_count:", step_count)
    if step_count % n_steps == 0:
        with torch.no_grad():
            next_obs_tensor = preprocess_obs(next_obs)
            last_val = model.policy.predict_values(next_obs_tensor)

        buffer.compute_returns_and_advantage(last_val, dones=done)

        model.policy.train()
        model.policy.optimizer.zero_grad()

        for rollout_data in buffer.get(batch_size=None):
            loss = compute_a2c_loss(model.policy, rollout_data)
            loss.backward()

        model.policy.optimizer.step()
        buffer.reset()


def save_model(path="a2c_satellite"):
    model.save(path)


def load_model(path="a2c_satellite"):
    global model
    model = A2C.load(path)
    model.set_env(dummy_env)
    return True


# def train_multiple_episodes(n_episodes=100):

#     for episode in range(n_episodes):
#         global step_count
#         obs = reset_env()
#         done = False
#         episode_reward = 0
#         while not done:
#             action = get_action(obs)
#             next_obs, reward, done = my_step(action)
#             store_transition(reward, done, next_obs)
#             obs = next_obs
#             print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Total Steps = {step_count}")
#             episode_reward += reward
#             if done:
#                 print("+++++++++++Episode finished+++++++++++++")
#                 break  # Optional, since the loop exits on `done` anyway
#         step_count = 0  # Reset step count for the next episode
#         print(f"Episode {episode + 1} finished with total reward: {episode_reward:.2f}")


# train_multiple_episodes(10)