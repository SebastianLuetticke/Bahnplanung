from manipulator_mujoco.envs import SimulationEnv
from stable_baselines3 import PPO
import numpy as np

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

# load a saved model in a new environment
models_dir = "models/"
model_path = f"{models_dir}/rl_simple"
log_dir = "logs"
env = SimulationEnv(TARGET_POINT,
                    origin_trajectory_simple, 0, 2,
                    algorithm="rl",
                    with_endpoint_prediction=False,
                    with_scaled_penalty=True,
                    should_orientation_vary=False,
                    should_x_vary=False,
                    orientation_offset=0,
                    x_offset=0,
                    render_mode="human")
model = PPO.load(model_path, env=env)

# execute a episode in the environment env with the loaded model
obs, info = env.reset()
terminated = False
total_reward = 0
while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
print(f"Total Cost: {-total_reward}")
env.close()