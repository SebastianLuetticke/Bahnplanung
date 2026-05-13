import numpy as np

def cma_norm(trajectory):
    new_trajectory = trajectory.copy()
    for i in range(len(new_trajectory)):
        quat_norm = np.linalg.norm(new_trajectory[i][3:7])
        for j in range(len(new_trajectory[i])):
            if j in [3, 4, 5, 6]: # [-1, 1]
                if quat_norm > 1e-6:
                    new_trajectory[i][j] /= quat_norm
                else:
                    if j == 6:
                        new_trajectory[i][j] = 1
                    else:
                        new_trajectory[i][j] = 0
            elif j == 0: # x: [-0.3, 0.5]
                new_trajectory[i][j] = new_trajectory[i][j] * 2.5 - 0.25
            elif j == 1: # y: [-0.6, -0.2]
                new_trajectory[i][j] = new_trajectory[i][j] * 5 + 2
            elif j == 2: # z: [0.1, 0.5]
                new_trajectory[i][j] = new_trajectory[i][j] * 5 - 1.5
            else: # gripper: [0, 1]
                new_trajectory[i][j] = new_trajectory[i][j] * 2 - 1
            
    return new_trajectory.flatten()

            
def cma_denorm(trajectory, n_waypoints, waypoint_dimension):
    new_trajectory = trajectory.copy()
    new_trajectory = new_trajectory.reshape((n_waypoints, waypoint_dimension))
    for i in range(len(new_trajectory)):
        quat_norm = np.linalg.norm(new_trajectory[i][3:7])
        for j in range(len(new_trajectory[i])):
            if j in [3, 4, 5, 6]: # [-1, 1], normieren!
                if quat_norm > 1e-6:
                    new_trajectory[i][j] /= quat_norm
                else:
                    if j == 6:
                        new_trajectory[i][j] = 1
                    else:
                        new_trajectory[i][j] = 0
            elif j == 0: # [-0.3, 0.3]
                new_trajectory[i][j] = (new_trajectory[i][j] + 0.25) * 0.4
            elif j == 1: # [-0.7, -0.2]
                new_trajectory[i][j] = (new_trajectory[i][j] - 2) * 0.2
            elif j == 2: # [0.1, 0.5]
                new_trajectory[i][j] = (new_trajectory[i][j] + 1.5) * 0.2
            else: # j = 7 -> [0, 1]
                new_trajectory[i][j] = (new_trajectory[i][j] + 1) * 0.5

    # add gripper strengths if needed
    if waypoint_dimension == 7:
        gripper_strength = np.full((n_waypoints, 1), 0.8)
        new_trajectory = np.hstack((new_trajectory, gripper_strength))
        
    return new_trajectory


def evaluate(trajectory, env):
    """
    Evaluates the trajectory by running it in an environment

    Args:
        trajectory (2D-list): list of configurations

    Returns:
        float: costs of the trajectory
    """
    
    env.reset()
    costs = 0
    terminated = False
    action_num = 0
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(trajectory[action_num])
        costs += reward
        action_num += 1
    
    return costs