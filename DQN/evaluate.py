import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from utils.env import CogSatEnv

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# set the seed
seed = 42

gymnasium.register(
    id='CogSatEnv-v1',  # Use the same ID here as you used in the script
    entry_point='env:CogSatEnv',
)

# Initialize the environment
env_id = "CogSatEnv-v1"
env = CogSatEnv()

env.reset(seed=seed)  # Reset the environment with the seed

dummy_env = DummyVecEnv([lambda: env])  # Wrap the environment with DummyVecEnv



epoch_length = 180 ## got through experiment
epoch_numbers = 100

total_steps = epoch_length * epoch_numbers

# Optional: Check the environment
check_env(env, warn=True)

# Instantiate the model
model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=10,
    batch_size=16,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10,
    verbose=1
)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
# measure perofmance of training

env.close()


