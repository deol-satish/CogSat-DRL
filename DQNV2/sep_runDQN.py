import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Import your custom environment
from your_env_module import CogSatEnv  # Replace with actual module path

# Create the environment
env = CogSatEnv()

# Optional: check if the environment follows Gym API
check_env(env)

# Create the DQN model
model = DQN(
    policy="MlpPolicy",     # You can try "CnnPolicy" if using image input
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=100,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_cogsat_tensorboard/"
)

# Train the model
model.learn(total_timesteps=50000)

# Save the model
model.save("dqn_cogsat")

# Optional: test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
