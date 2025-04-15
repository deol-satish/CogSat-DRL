import gymnasium
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env import LeoGeoEnv

# set the seed
seed = 42

gymnasium.register(
    id='LeoGeoEnv-v3.1',  # Use the same ID here as you used in the script
    entry_point='env:LeoGeoEnv',
)

# Initialize the environment
env_id = "LeoGeoEnv-v3.1"
env = make_vec_env(env_id, n_envs=1, seed=seed)

epoch_length = 884 ## got through experiment
epoch_numbers = 100

# Set up the checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')

# Specify the policy network architecture, here we are using the default MIP
model = A2C("MultiInputPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log="./a2c_leogeo_tensorboard/",
            seed=seed, learning_rate=0.0001)

# Define the total number of timesteps to train the model
total_timesteps = epoch_length*epoch_numbers

# Train the model
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Save the model
model.save("a2c_leogeoenv_1")

env.close()

# If you want to load the saved model, you can use the following:
# model = A2C.load("a2c_leogeoenv", env=env)


###########
# import gymnasium
# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
# from env import LeoGeoEnv
#
# # Define the LinearLearningRateSchedule callback
# class LinearLearningRateSchedule(BaseCallback):
#     def __init__(self, initial_lr, final_lr, total_timesteps, verbose=0):
#         super(LinearLearningRateSchedule, self).__init__(verbose)
#         self.initial_lr = initial_lr
#         self.final_lr = final_lr
#         self.total_timesteps = total_timesteps
#
#     def _on_step(self) -> bool:
#         # Calculate the current learning rate
#         progress = self.num_timesteps / self.total_timesteps
#         current_lr = self.initial_lr + progress * (self.final_lr - self.initial_lr)
#         # Update the learning rate
#         self.model.lr_schedule = current_lr
#         return True
#
# # Set the seed
# seed = 42
#
# gymnasium.register(
#     id='LeoGeoEnv-v3.1',
#     entry_point='env:LeoGeoEnv',
# )
#
# # Initialize the environment
# env_id = "LeoGeoEnv-v3.1"
# env = make_vec_env(env_id, n_envs=1, seed=seed)
#
# epoch_length = 884  # Got through experiment
# epoch_numbers = 1500
#
# # Set up the checkpoint callback
# checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')
#
# # Define initial and final learning rates
# initial_lr = 0.0007
# final_lr = 0.00001
#
# # Specify the policy network architecture, here we are using the default MLP
# model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./a2c_leogeo_tensorboard/", seed=seed, learning_rate=initial_lr)
#
# # Define the total number of timesteps to train the model
# total_timesteps = epoch_length * epoch_numbers
#
# # Set up the linear learning rate schedule callback
# lr_schedule_callback = LinearLearningRateSchedule(initial_lr, final_lr, total_timesteps)
#
# # Train the model with the learning rate schedule and checkpoint callbacks
# model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, lr_schedule_callback])
#
# # Save the model
# model.save("a2c_leogeoenv_1")
#
# env.close()
#
# # If you want to load the saved model, you can use the following:
# # model = A2C.load("a2c_leogeoenv", env=env)






# import gymnasium
# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.callbacks import CheckpointCallback
# from env import LeoGeoEnv
#
# # set the seed
# seed = 42
#
# gymnasium.register(
#     id='LeoGeoEnv-v3.1',  # Use the same ID here as you used in the script
#     entry_point='env:LeoGeoEnv',
# )
#
# # Initialize the environment
# env_id = "LeoGeoEnv-v3.1"
# env = make_vec_env(env_id, n_envs=1, seed=seed)
#
# epoch_length = 884 ## got through experiment
# epoch_numbers = 1500
#
# # Set up the checkpoint callback
# checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')
#
# # Specify the policy network architecture, here we are using the default MLP
# model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./a2c_leogeo_tensorboard/", seed=seed, learning_rate=0.0001)
#
# # Define the total number of timesteps to train the model
# total_timesteps = epoch_length*epoch_numbers
#
# # Train the model
# model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
#
# # Save the model
# model.save("a2c_leogeoenv_1")
#
# env.close()
#
# # If you want to load the saved model, you can use the following:
# # model = A2C.load("a2c_leogeoenv", env=env)
