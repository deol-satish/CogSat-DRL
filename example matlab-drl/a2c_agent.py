import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

# === Setup Pendulum environment ===
env = gym.make("Pendulum-v1")
dummy_env = DummyVecEnv([lambda: env])

model = A2C("MlpPolicy", dummy_env, verbose=0, device="cpu")

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

def reset_env():
    global obs_last
    obs_last = dummy_env.reset()
    return obs_last

def get_action(obs):
    global obs_last, action_last, value_last, log_prob_last
    obs_last = np.array(obs, dtype=np.float32).reshape((1, -1))
    obs_tensor = torch.tensor(obs_last).float().to(model.device)
    with torch.no_grad():
        action_tensor, value_tensor, log_prob_tensor = model.policy.forward(obs_tensor)
    action_last = action_tensor
    value_last = value_tensor
    log_prob_last = log_prob_tensor
    return action_tensor.cpu().numpy()[0]

def my_step(action):
    next_obs, reward, terminated, truncated, _ = env.step(np.array(action))
    done = terminated or truncated
    return next_obs, reward, done

def store_transition(reward, done, next_obs):
    global step_count, obs_last, action_last, value_last, log_prob_last
    reward = np.array([reward], dtype=np.float32)
    done = np.array([done], dtype=bool)
    next_obs = np.array(next_obs, dtype=np.float32).reshape((1, -1))
    buffer.add(obs_last, action_last, reward, done, value_last, log_prob_last)
    step_count += 1
    obs_last = next_obs
    if step_count % n_steps == 0:
        with torch.no_grad():
            last_val = model.policy.predict_values(torch.tensor(next_obs).float().to(model.device))
        buffer.compute_returns_and_advantage(last_val, dones=done)
        model.train()

def save_model(path="a2c_pendulum"):
    model.save(path)

def load_model(path="a2c_pendulum"):
    global model
    model = A2C.load(path)
    model.set_env(dummy_env)
    return True
