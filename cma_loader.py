from manipulator_mujoco.envs import SimulationEnv
from cma_utils import cma_norm, cma_denorm, evaluate
import numpy as np
import pandas as pd
import ast


N_WAYPOINTS = 3
WAYPOINT_DIMENSION = 7 # x, y, z, q1, q2, q3, q4, (gripper)
TARGET_POINT = [0.3, -0.4, 0.13, 0, 0, 0, 1]
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

filename = "cma_data/cma26_simple.csv"
generation = 1000

# read trajectory from csv-file
df = pd.read_csv(filename)
row = df[df["generation"] == generation]
trajectory_str = row["best_x"].iloc[0]

# convert to numpy array
normed_trajectory = np.array(ast.literal_eval(trajectory_str))
# denorm the 1D-trajectory-list to a 2D-trajectory-list of configurations
trajectory = cma_denorm(normed_trajectory, N_WAYPOINTS, WAYPOINT_DIMENSION)

env = SimulationEnv(TARGET_POINT,
                    origin_trajectory_simple, 0, 2,
                    algorithm="cma",
                    with_endpoint_prediction=False,
                    with_scaled_penalty=True,
                    should_orientation_vary=False,
                    should_x_vary=False,
                    orientation_offset=0,
                    x_offset=0,
                    render_mode="human")

print(f"Total Cost: {evaluate(trajectory, env)}")
env.close()