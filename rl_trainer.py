from manipulator_mujoco.envs import SimulationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np


TIMESTEPS_TO_SAVE = 12288 # specifies after how many steps a model is saved
TARGET_POINT = [0.3, -0.4, 0.13, 0, 0, 0, 1]
ALGORITHM_NAME = "PPO"
origin_trajectory_simple = np.array([
            [-0.2, -0.55, 0.25, 0, 0, 0, 1],
            [0.05, -0.475, 0.45, 0, 0, 0, 1],
            [0.3, -0.4, 0.25, 0, 0, 0, 1]
        ])
origin_trajectory_hard = np.array([
            [-0.2, -0.55, 0.25, 0, 0, 0, 1, 0.8],
            [0.3, -0.4, 0.25, 0, 0, 0, 1, 0.8],
            [0.3, -0.4, 0.101, 0, 0, 0, 1, 0]
        ])


models_dir = "new_models/"
log_dir = "new_logs"
# Creates the storage folders if they do not yet exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def make_env():
    """
    Creates a new environment.

    Returns:
        SimulationEnv: the new environment
    """
    env = SimulationEnv(TARGET_POINT,
                    origin_trajectory_simple, 0, 2,
                    algorithm="rl",
                    with_endpoint_prediction=False,
                    with_scaled_penalty=True,
                    should_orientation_vary=False,
                    should_x_vary=False,
                    orientation_offset=0,
                    x_offset=0,
                    render_mode="rgb_array")
    return env


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    n_envs = 6
    # Create n_envs parallel environments
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", n_steps=2048, tensorboard_log=log_dir)
    
    for i in range(1,10000000000): # for manual abortion
        model.learn(total_timesteps=TIMESTEPS_TO_SAVE, reset_num_timesteps=False)
        model.save(f"{models_dir}/{TIMESTEPS_TO_SAVE*i}")
