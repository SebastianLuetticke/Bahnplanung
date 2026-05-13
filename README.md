# Comparison of Optimization and Reinforcement Learning for Path Planning in Robotics

This repository contains the code for the master's thesis with the title mentioned above. It is based on Ian Chaung's project https://github.com/ian-chuang/Manipulator-Mujoco.

A scenario has been implemented in which a gripper robot is tasked with transporting a sausage from a starting platform to a target platform.
An optimization agent and a reinforcement learning agent can be trained for this problem.

---

## Installation

Navigate to the repository's main folder and run the following:

```bash
pip install -e .
pip install stable-baselines3[extra]
pip install cma
````

## Testing and Training an Agent
Settings for the simulation can be configured in the `SimulationEnv()` constructor:
* `sausage_end_position`: Where should the sausage's target point be?
* `origin_trajectory`: The origin trajectory that the agents can use for orientation
* `starting_waypoint`: The waypoint at which the trajectory starts, typically 0
* `ending_waypoint`: The waypoint at which the trajectory ends
* `algorithm`: Either `"cma"` or `"rl"`
* `with_endpoint_prediction`: Should the sausage's endpoint be part of the trajectory?
* `with_scaled_penalty`: Should the distance-based costs be scaled?
* `should_orientation_vary`: Should the sausage’s starting orientation vary?
* `should_x_vary`: Should the x-coordinate of the obstacle position vary?
* `orientation_offset`: Fixed value that rotates the sausage at the start
* `x_offset`: Fixed value that shifts the obstacle
* `render_mode`: Either `"human"` for with GUI or `"rgb_array"` for without GUI


### Optimization Agent
In the files, the agent to be loaded or trained can be selected using `filename`.  

If the difficult version of the problem is to be simulated, `WAYPOINT_DIMENSION = 8` must be set; otherwise, `WAYPOINT_DIMENSION = 7`. In addition, the corresponding origin trajectory must be adjusted.

#### Test the Agent
Run `cma_loader.py`
#### Train the Agent
Run `cma_trainer.py`


### Reinforcement Learning Agent
In the file, you can use `model_path` to select the RL agent model to be loaded or trained.  

Depending on the difficulty version to be simulated, the corresponding origin trajectory must be adjusted.

#### Test the Agent
Run `rl_loader.py`
#### Train the Agent
Run `rl_trainer.py`

---

Depending on your environment setup, the costs in the results may vary.
