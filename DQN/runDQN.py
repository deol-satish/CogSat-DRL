import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

# Import your custom environment
from utils.env import CogSatEnv  # Assuming it's in cogsat_env.py

seed = 42

gym.register(
    id='CogSatEnv-v1',  # Use the same ID here as you used in the script
    entry_point='env:CogSatEnv',
)


# Initialize the environment
env_id = "CogSatEnv-v1"
env = make_vec_env(env_id, n_envs=1, seed=seed)

epoch_length = 884 ## got through experiment
epoch_numbers = 100

# Optional: Check the environment
check_env(env, warn=True)

# Instantiate the model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=1000)

# Save the model
model.save("dqn_cogsat")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
