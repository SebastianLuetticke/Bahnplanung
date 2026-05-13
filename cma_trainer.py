from manipulator_mujoco.envs import SimulationEnv
from cma_utils import cma_norm, cma_denorm, evaluate
import cma
import csv
import os
import pickle
import numpy as np


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


env = SimulationEnv(TARGET_POINT,
                    origin_trajectory_simple, 0, 2,
                    algorithm="cma",
                    with_endpoint_prediction=False,
                    with_scaled_penalty=True,
                    should_orientation_vary=False,
                    should_x_vary=False,
                    orientation_offset=0,
                    x_offset=0,
                    render_mode="rgb_array")

sigma_init = 0.1
es = cma.CMAEvolutionStrategy(cma_norm(origin_trajectory_simple), sigma_init, {'popsize': 26, "bounds": [-1, 1]})

filename = "new_cma_evaluation"
logfile = f"{filename}.csv"
file_exists = os.path.isfile(logfile)

csv_file = open(logfile, "a", newline="")
writer = csv.writer(csv_file)
if not file_exists:
    # construction of the csv-file
    writer.writerow([
        "generation",
        "evals",
        "sigma",
        "best_f",
        "best_x",
        "mean",
        "best_f_gen",
        "best_x_gen",
        "f_gen_min",
        "f_gen_mean",
        "f_gen_std",
    ])
    csv_file.flush()

# run cma algorithm until manual abortion
try:
    while True:
        solutions = es.ask()
        costs = [evaluate(cma_denorm(s, N_WAYPOINTS, WAYPOINT_DIMENSION), env) for s in solutions]
        es.tell(solutions, costs)
        es.disp()
        es.logger.add()
        
        fitnesses = np.array(costs)
        best_idx = np.argmin(fitnesses)
        best_f_gen = fitnesses[best_idx]
        best_x_gen = solutions[best_idx]

        writer.writerow([
            es.countiter,
            es.countevals,
            es.sigma,
            es.best.f,
            es.best.x.tolist(),
            es.mean.tolist(),
            best_f_gen,
            best_x_gen.tolist(),
            np.min(fitnesses),
            np.mean(fitnesses),
            np.std(fitnesses)
        ])
        csv_file.flush()
            
except KeyboardInterrupt:
    with open(f"{filename}_checkpoint.pkl", "wb") as f:
        pickle.dump(es, f)
    print("Optimierung manuell gestoppt.")
finally:
    csv_file.close()


env.close()