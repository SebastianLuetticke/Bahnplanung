import os
import time
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, AuboI5, AG95
from manipulator_mujoco.props import Primitive, Sausage, Box, Site
from manipulator_mujoco.controllers import OperationalSpaceController
from manipulator_mujoco.utils.controller_utils import (
    pose_error,
)
from manipulator_mujoco.utils.transform_utils import (
    mat2euler, euler2mat, quat2mat, mat2quat
)
from dm_control.rl.control import PhysicsError
from scipy.spatial.transform import Rotation
import random


BREAK_THRESHOLD = None
CLOSENESS_THRESHOLD = 0.01

OBSTACLE_SIZE = [0.03, 0.03, 0.1]
OBSTACLE_POS = [-0.03, -0.35, 0.1]
OBSTACLE_CORNER1 = [
    OBSTACLE_POS[0]-OBSTACLE_SIZE[0],
    OBSTACLE_POS[1]-OBSTACLE_SIZE[1],
    OBSTACLE_POS[2]+OBSTACLE_SIZE[2]
    ]
OBSTACLE_CORNER2 = [
    OBSTACLE_POS[0]+OBSTACLE_SIZE[0],
    OBSTACLE_POS[1]-OBSTACLE_SIZE[1],
    OBSTACLE_POS[2]+OBSTACLE_SIZE[2]
    ]
OBSTACLE_CORNER3 = [
    OBSTACLE_POS[0]+OBSTACLE_SIZE[0],
    OBSTACLE_POS[1]+OBSTACLE_SIZE[1],
    OBSTACLE_POS[2]+OBSTACLE_SIZE[2]
    ]
OBSTACLE_CORNER4 = [
    OBSTACLE_POS[0]-OBSTACLE_SIZE[0],
    OBSTACLE_POS[1]+OBSTACLE_SIZE[1],
    OBSTACLE_POS[2]+OBSTACLE_SIZE[2]
    ]

START_PLATFORM_SIZE = [0.22, 0.07, 0.05]
START_PLATFORM_POS = [-0.3, -0.3, 0.05]
START_PLATFORM_CORNER1 = [
    START_PLATFORM_POS[0]+START_PLATFORM_SIZE[1],
    START_PLATFORM_POS[1]-START_PLATFORM_SIZE[0],
    START_PLATFORM_POS[2]+START_PLATFORM_SIZE[2]
    ]
START_PLATFORM_CORNER2 = [
    START_PLATFORM_POS[0]+START_PLATFORM_SIZE[1],
    START_PLATFORM_POS[1]+START_PLATFORM_SIZE[0],
    START_PLATFORM_POS[2]+START_PLATFORM_SIZE[2]
    ]

END_PLATFORM_SIZE = [0.22, 0.07, 0.05]
END_PLATFORM_POS = [0.3, -0.4, 0.05]
END_PLATFORM_CORNER1 = [
    END_PLATFORM_POS[0]-END_PLATFORM_SIZE[0],
    END_PLATFORM_POS[1]-END_PLATFORM_SIZE[1],
    END_PLATFORM_POS[2]+END_PLATFORM_SIZE[2]
    ]
END_PLATFORM_CORNER2 = [
    END_PLATFORM_POS[0]-END_PLATFORM_SIZE[0],
    END_PLATFORM_POS[1]+END_PLATFORM_SIZE[1],
    END_PLATFORM_POS[2]+END_PLATFORM_SIZE[2]
    ]


class SimulationEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(self, sausage_end_position, origin_trajectory, starting_waypoint, ending_waypoint, algorithm="cma", with_endpoint_prediction=False, with_scaled_penalty=True, should_orientation_vary=False, should_x_vary=False, orientation_offset=0, x_offset=0, render_mode=None):
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(124,), dtype=np.float32
        )

        action_dimension = origin_trajectory.shape[1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dimension,), dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode
        
        self._algorithm = algorithm
        self._starting_waypoint = starting_waypoint
        self._ending_waypoint = ending_waypoint
        self._with_endpoint_prediction = with_endpoint_prediction
        self._with_scaled_penalty = with_scaled_penalty

        ############################
        # create MJCF model
        ############################
        
        # checkerboard floor
        self._arena = StandardArena()

        # ur5e arm
        self._arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        
        # ag95 gripper
        self._gripper = AG95()

        # attach gripper to arm
        self._arm.attach_tool(self._gripper.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0])

        # sausage to be manipulated
        self._sausage = Sausage()
        
        # obstacle between start- and endplatform
        self._obstacle = Box(
            colliders_on=[True, True, True, True, False, True],
            name="obstacle",
            type="box",
            size=OBSTACLE_SIZE,
            pos=OBSTACLE_POS,
            quat=[1, 0, 0, 0],
            rgba=[0, 1, 0, 1]
            )
        
        # box to raise the robot arm
        self._robot_platform = Box(
            colliders_on=[False]*6,
            name="robot_platform",
            type="box",
            size=[0.08, 0.08, 0.05],
            pos=[0, 0, 0.05],
            quat=[1, 0, 0, 0],
            rgba=[0.25, 0.25, 0.25, 1]
            )
        
        # startplatform for the sausage
        self._start_platform = Box(
            colliders_on=[True, True, True, True, False, False],
            name="start_platform",
            type="box",
            size=START_PLATFORM_SIZE,
            pos=START_PLATFORM_POS,
            quat=[0.7071068, 0, 0, 0.7071068],
            rgba=[0, 1, 0, 1]
            )
        
        # endplatform for the sausage
        self._end_platform = Box(
            colliders_on=[True, True, True, True, False, False],
            name="end_platform",
            type="box",
            size=END_PLATFORM_SIZE,
            pos=END_PLATFORM_POS,
            quat=[1, 0, 0, 0],
            rgba=[0, 1, 0, 1]
            )
        
        
        # optional: site to mark a certain point in the scene
        #self._site1 = Site(
        #    size=[0.03, 0.03],
        #    pos=START_PLATFORM_CORNER1,
        #    rgba=[1,0,0,1]
        #)
        #self._arena.attach(
        #    self._site1.mjcf_model, pos=[0,0,0]
        #)
        
        # attach robot platform to arena
        self._arena.attach(
            self._robot_platform.mjcf_model, pos=[0,0,0]
        )
        # attach start platform to arena
        self._arena.attach(
            self._start_platform.mjcf_model, pos=[0,0,0]
        )
        # attach end platform to arena
        self._arena.attach(
            self._end_platform.mjcf_model, pos=[0,0,0]
        )
        # attach robot arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0.1]
        )
        # attach sausage to arena
        self._sausage_wrapper = self._arena.attach_free(
            self._sausage.mjcf_model, pos=[-0.3, -0.3, 0.12], quat=[0.7071068, 0, 0, 0.7071068]
        )
        # attach obstacle to arena
        self._arena.attach_free(
            self._obstacle.mjcf_model, pos=[0,0,0]
        )
       
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # set up OSC controller
        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None
        
        self._should_orientation_vary = should_orientation_vary
        self._should_x_vary = should_x_vary
        self._orientation_offset = orientation_offset
        self._x_offset = x_offset
        
        self._sausage_end_position = sausage_end_position
        self._is_sausage_broken = False
        self._is_movement_finished = False
        self._is_action_finished = False
        self._simulation_steps = 0
        self._total_steps = 0
        self._origin_trajectory = origin_trajectory
        self._waypoints = len(origin_trajectory)
        self._current_waypoint = 0
        self._collision_handled = False
        self._sausage_velocities = np.zeros((7,3))
        
        self._sausage_geom_names = [
            "sausage/seg0_geom",
            "sausage/seg1_geom",
            "sausage/seg2_geom",
            "sausage/seg3_geom",
            "sausage/seg4_geom",
            "sausage/seg5_geom",
            "sausage/seg6_geom",
        ]
        self._sausage_geom_ids = [
            self._physics.model.name2id(name, "geom")
            for name in self._sausage_geom_names
        ]
        
        self._gripper_id = self._physics.model.name2id("ur5e/dh_ag95_gripper/fingers_actuator", "actuator")


    def _get_obs(self) -> np.ndarray:
        """
        Calculates all importand features of the current environment state.

        Returns:
            np.ndarray: 124-dimensional array of real values
        """
        gripper_pose = self.get_gripper_position(True)
        gripper_pos = gripper_pose[:3]
        sausage_left_pos = self.get_sausage_position("left", False)
        sausage_middle_pose = self.get_sausage_position("middle", True)
        sausage_middle_pos = sausage_middle_pose[:3]
        sausage_right_pos = self.get_sausage_position("right", False)
        sausage_end_error = pose_error(self.ignore_x_rotation(sausage_middle_pose), self._sausage_end_position)
        
        observations = np.array([])
        ### distances to start_platform
        ## corner1
        # 0-2 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - START_PLATFORM_CORNER1])
        # 3-5 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - START_PLATFORM_CORNER1])
        # 6-8 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - START_PLATFORM_CORNER1])
        # 9-11 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - START_PLATFORM_CORNER1])
        ## corner2
        # 12-14 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - START_PLATFORM_CORNER2])
        # 15-17 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - START_PLATFORM_CORNER2])
        # 18-20 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - START_PLATFORM_CORNER2])
        # 21-23 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - START_PLATFORM_CORNER2])
        
        ### distances to obstacle
        ## corner1
        # 24-26 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - OBSTACLE_CORNER1])
        # 27-29 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - OBSTACLE_CORNER1])
        # 30-32 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - OBSTACLE_CORNER1])
        # 33-35 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - OBSTACLE_CORNER1])
        ## corner2
        # 36-38 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - OBSTACLE_CORNER2])
        # 39-41 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - OBSTACLE_CORNER2])
        # 42-44 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - OBSTACLE_CORNER2])
        # 45-47 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - OBSTACLE_CORNER2])
        ## corner3
        # 48-50 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - OBSTACLE_CORNER3])
        # 51-53 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - OBSTACLE_CORNER3])
        # 54-56 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - OBSTACLE_CORNER3])
        # 57-59 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - OBSTACLE_CORNER3])
        ## corner4
        # 60-62 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - OBSTACLE_CORNER4])
        # 63-65 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - OBSTACLE_CORNER4])
        # 66-68 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - OBSTACLE_CORNER4])
        # 69-71 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - OBSTACLE_CORNER4])
        
        ### distances to end_platform
        ## corner1
        # 72-74 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - END_PLATFORM_CORNER1])
        # 75-77 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - END_PLATFORM_CORNER1])
        # 78-80 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - END_PLATFORM_CORNER1])
        # 81-83 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - END_PLATFORM_CORNER1])
        ## corner2
        # 84-86 x,y,z gripper-end effector
        observations = np.concatenate([observations, gripper_pos - END_PLATFORM_CORNER2])
        # 87-89 x,y,z sausage_site_left
        observations = np.concatenate([observations, sausage_left_pos - END_PLATFORM_CORNER2])
        # 90-92 x,y,z sausage_site_middle
        observations = np.concatenate([observations, sausage_middle_pos - END_PLATFORM_CORNER2])
        # 93-95 x,y,z sausage_site_right
        observations = np.concatenate([observations, sausage_right_pos - END_PLATFORM_CORNER2])
        
        ### 96-101 distances x,y,z,orientation difference form end_position to sausage_site_middle
        observations = np.concatenate([observations, sausage_end_error])
        
        ### 102-122 sausage segment velocities
        observations = np.concatenate([observations, self._sausage_velocities.flatten()])
        
        ### 123 waypoint
        observations = np.concatenate([observations, [self._current_waypoint]])
        
        return np.float32(observations)
    
    
    def _get_reward(self, info, algorithm, with_endpoint_prediction, with_scaled_penalty):
        """
        Calculates the reward for a simulation step.
        
        Args:
            info (dict): information of the current simulation step
            algorithm (String): "cma" or "rl"
            with_endpoint_prediction (bool): is the endpoint part of the predicted trajectory?
            with_scaled_penalty (bool): should the distance penalty be scaled?

        Returns:
            np.ndarray: 124-dimensional array of real values
        """
        cost = info["break_value"] * self._timestep
        
        action_over = info["action_finished"] or self._simulation_steps > 1000
        
        if with_endpoint_prediction:
            if action_over:
                sausage_position = self.ignore_x_rotation(self.get_sausage_position())
                # reduce the costs for getting closer to the target position
                if self._current_waypoint >= self._starting_waypoint and self._current_waypoint <= self._ending_waypoint:
                    current_distance = np.linalg.norm(pose_error(self._sausage_end_position, sausage_position))
                    distance_improvement = self.prev_distance - current_distance
                    self.prev_distance = current_distance
                    if with_scaled_penalty:
                        cost -= distance_improvement / 1.40 * 30
                    else:
                        cost -= distance_improvement / 1.40
                    
            # end of trajectory
            if self._current_waypoint == self._waypoints and action_over:
                sausage_position = self.get_sausage_position()
                total_distance = np.linalg.norm(pose_error(self._sausage_end_position, self.ignore_x_rotation(sausage_position)))
                # penalty if sausage is not naer enough to the target
                if total_distance >= 0.01:
                    if with_scaled_penalty:
                        cost += 30 + total_distance / 1.40 * 30
                    else:
                        cost += 30 + total_distance / 1.40
                          
        # penalty for the first collision in the trajectory
        if info["is_colliding"] and not self._collision_handled:
            distance_to_box_center = 1 / np.linalg.norm(info["collision_point"] - info["collision_box_position"])
            self._collision_handled = True
            if with_scaled_penalty:
                cost += 30 + distance_to_box_center / 32.36 * 30
            else:
                cost += 30 + distance_to_box_center / 32.36
        
        # penalty if the point was unreachable
        if self._simulation_steps > 1000:
            if with_scaled_penalty:
                cost += 30 + info["total_movement_error"] / 1.05 * 30
            else:
                cost += 30 + info["total_movement_error"] / 1.05
        
        # positive costs for minimizing cma, negative rewards for rl
        if algorithm == "cma":
            return cost
        return -cost

    
    def calculate_break_metric(self):
        """
        Calculates the break value in the current simulation step.

        Returns:
            float: break value in the current simulation step.
        """
        
        break_value = 0 
        
        for i in range(0, 6):
            # get the current velocities from the engine
            v_i = self._physics.data.cvel[self._physics.model.name2id(f"sausage/seg{i}", "body")][3:]
            v_i1 = self._physics.data.cvel[self._physics.model.name2id(f"sausage/seg{i+1}", "body")][3:]
            # calculate the acceleration
            a_i = (v_i - self._sausage_velocities[i]) / self._timestep
            a_i1 = (v_i1 - self._sausage_velocities[i+1]) / self._timestep
            diff = np.linalg.norm(a_i1 - a_i)
            break_value += diff
            
            # update the current velocities
            self._sausage_velocities[i] = v_i.copy()
        
        # update the velocity of the last segment because this doesn't happen in the loop
        self._sausage_velocities[6] = self._physics.data.cvel[self._physics.model.name2id(f"sausage/seg{6}", "body")][3:].copy()
        
        return break_value
    
    
    def _get_info(self, total_movement_error=0) -> dict:
        """
        Determines the relevant information for the current simulation step.
        
        Args:
            total_movement_error: the total error from the current robot ar position to the target point

        Returns:
            dict: all relevant information for a simulation step
        """
        
        # If BREAK_THREASHOLD is defined, mark the sausage red if the break value is too high.
        break_value = self.calculate_break_metric()
        if BREAK_THRESHOLD != None and not self._is_sausage_broken:
            if break_value > BREAK_THRESHOLD:
                for geom_id in self._sausage_geom_ids:
                    self._physics.model.geom_rgba[geom_id] = np.array([1.0, 0.0, 0.0, 1.0])
                    self._is_sausage_broken = True
        
        info = {
            "action_finished": self._is_action_finished,
            "total_movement_error": total_movement_error,
            "sausage_broken": self._is_sausage_broken,
            "is_colliding": False,
            "collision_point": None,
            "collision_box_position": None,
            "break_value": break_value
            }
        
        # check whether the robot or the sausage collides with a box
        is_colliding, collision_point, geom1_id, geom2_id, geom1_name, geom2_name = self.is_colliding(
            [("ur5e"), ("sausage")],
            [
                (
                    "obstacle_model",
                    "start_platform_model",
                    "end_platform_model"
                ), (
                    "obstacle_model/obstacle_collision",
                    "start_platform_model/start_platform_collision",
                    "end_platform_model/end_platform_collision"
                )
            ],
        )
        
        # if a collision happened, update the collision information
        if is_colliding:
            info["is_colliding"] = True
            info["collision_point"] = collision_point
            
            if geom1_name.startswith(("obstacle", "start_platform", "end_platform")):
                info["collision_box_position"] = self.get_box_position(geom1_name.partition("_model")[0])
            else:
                info["collision_box_position"] = self.get_box_position(geom2_name.partition("_model")[0])
        
        return info


    def reset(self, seed=None, options=None, ) -> tuple:
        """
        Executes the reset method.
        
        Args:
            seed (int): to start the simulation with a certain initializationof the random number generator

        Returns:
            tuple: the observation of the current environment state and the information
        """
        
        super().reset(seed=seed)
        
        #optional: Vary the sausage orientation and obstacle position.
        orientation_offset = 90 + self._orientation_offset
        if self._should_orientation_vary:
            orientation_offset += random.uniform(-20, 20)
        rotation_offset = Rotation.from_euler('z', orientation_offset, degrees=True)
        radians_offset = np.deg2rad(orientation_offset)
        rotation_quaternion = rotation_offset.as_quat() # Quaternion (x, y, z, w)
        x_offset = self._x_offset
        if self._should_x_vary:
            x_offset += random.uniform(-0.165, 0.075)
        
        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos = [2.03, -1.11, 1.76, 0.923, 1.57, -radians_offset+0.46] # at sausage
            # rotate sausage
            self._physics.data.joint("sausage/").qpos[3:] = [rotation_quaternion[3], rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2]]
            # shift obstacle
            self._physics.data.joint("obstacle_model/").qpos[:3] = [x_offset, 0, 0]
            
        self._is_sausage_broken = False
        self._is_movement_finished = False
        self._is_action_finished = False
        self._simulation_steps = 0
        self._total_steps = 0
        self._current_waypoint = 0
        self._collision_handled = False
        
        # execute a pre-step to grab the sausage
        arm_start_point = [-0.3, -0.3, 0.101, rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2], rotation_quaternion[3], 0.8]
        self.help_step(arm_start_point, in_reset=True)
        
        sausage_position = self.ignore_x_rotation(self.get_sausage_position())
        self.prev_distance = np.linalg.norm(pose_error(self._sausage_end_position, sausage_position))
        
        for i in range(7):
            self._sausage_velocities[i] = self._physics.data.cvel[self._physics.model.name2id(f"sausage/seg{i}", "body")][3:]
        

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def step(self, action: np.ndarray) -> tuple:
        """
        Executes the step method.
        
        Args:
            action (np.ndarray): real values to determine the action to execute
                x-, y-, z-coordinate, four quaternion values, optional: grip strength

        Returns:
            tuple: the observation of the current environment state,
                the reward for the executed action,
                bool if the simulation is terminated,
                bool if the simulation is truncated
        """
        reward = 0.0
        
        try:
            real_action = action
            if self._algorithm == "rl":
                real_action = self.transform_action(real_action, self._with_endpoint_prediction)
            
            reward += self.help_step(real_action)
            self._current_waypoint += 1
            
            if self._current_waypoint > self._ending_waypoint:
                # move the gripper to specified end point
                ending_points = [
                    [0.3, -0.4, 0.101, 0, 0, 0, 1, 0.8], # end gripper point
                    [0.3, -0.4, 0.101, 0, 0, 0, 1, 0]
                ]
                if self._with_endpoint_prediction:
                    # only lifts the gripper a bit from the predicted end point
                    ending_points = [
                        [real_action[0], real_action[1], real_action[2]+0.2, real_action[3], real_action[4], real_action[5], real_action[6], 0]
                    ]
                for point in ending_points:
                    reward += self.help_step(point)
                    
            observation = self._get_obs()
            info = {}
            truncated = False
            terminated = self._current_waypoint == self._waypoints
            
        except PhysicsError:
            reward = -100.0
            observation, info = self.reset()
            truncated = False
            terminated = True
        
        return observation, np.float32(reward), terminated, truncated, info
        
        
    def help_step(self, action, in_reset=False):
        """
        Controls the action execution.
        
        Args:
            action (np.ndarray): real values to determine the action to execute
                x-, y-, z-coordinate, four quaternion values, optional: grip strength
            in_reset (bool): If the method is called in the reset step, then it is a pre-step

        Returns:
            float: the reward of the whole action execution
        """
        
        self._is_action_finished = False
        reward = 0.0
        self._simulation_steps = 0
        
        # Check if the action is finished or the point was unreachable
        while(not self._is_action_finished and self._simulation_steps <= 1000):
            total_movement_error = self.simulation_step(action)
            
            if not in_reset:
                # This things are only needed in real steps executions
                self._simulation_steps += 1
                self._total_steps += 1
                info = self._get_info(total_movement_error)
                reward += self._get_reward(info, self._algorithm, self._with_endpoint_prediction, self._with_scaled_penalty)
             
        return reward
    
    
    def simulation_step(self, action):
        """
        Executes a simulation step.
        
        Args:
            action (np.ndarray): real values to determine the action to execute
                x-, y-, z-coordinate, four quaternion values, grip strength

        Returns:
            float: the movement_error to the target point
        """
        
        # run OSC controller to move to target pose
        movement_errors  = self._controller.run(action[:7])
        total_movement_error = np.linalg.norm(movement_errors)
        # the movement is finished if the gripper is near enough the target point
        self._is_movement_finished = total_movement_error < CLOSENESS_THRESHOLD
        
        if(self._is_movement_finished):
            # increase the grip strength step for step
            distance_force = self._physics.data.ctrl[self._gripper_id] - action[7]
            if distance_force < 0:
                self._physics.data.ctrl[self._gripper_id] += min(abs(distance_force), 0.01)
            elif distance_force > 0:
                self._physics.data.ctrl[self._gripper_id] -= min(abs(distance_force), 0.01)
            else:
                self._is_action_finished = True
        
        # update the physics
        self._physics.step()
        
        # render frame
        if self._render_mode == "human":
            self._render_frame()
            
        return total_movement_error
    
    
    def transform_action(self, action, with_endpoint_prediction):
        """
        Transforms the predicted action form the rl-agnet to a valid action.
        
        Args:
            action (np.ndarray): real values to determine the action to execute
                x-, y-, z-coordinate, four quaternion values, optional: grip strength
            with_endpoint_prediction (bool): if the endpoint is included in the action

        Returns:
            (np.ndarray): the transformed action
        """
        
        new_action = action.copy()
        # norm the quaternion to norm 1
        quat_norm = np.linalg.norm(new_action[3:7])
        for i in range(len(action)):
            if i in [3, 4, 5, 6]:
                if quat_norm > 1e-6:
                    new_action[i] /= quat_norm
                else:
                    if i == 6:
                        new_action[i] = 1
                    else:
                        new_action[i] = 0
            else:
                # add the action offset to the coordinates (and the grip strength)
                new_action[i] = self._origin_trajectory[self._current_waypoint][i] + (new_action[i] / 5)
        
        if not with_endpoint_prediction:
            # append a grip strength
            new_action = np.append(new_action, 0.8)
        
        return new_action
    
    
    def ignore_x_rotation(self, pose):
        """
        Deletes the x-Rotation of a pose.
        
        Args:
            pose (np.ndarray): x-, y-, z-coordinate + 4 quaternion values

        Returns:
            (np.ndarray): the new pose without x-Rotation
        """
        
        new_pose = pose.copy()
        euler = mat2euler(quat2mat(new_pose[3:]))
        euler[0] = 0
        new_quat = mat2quat(euler2mat(euler))
        
        new_pose[3] = new_quat[0]
        new_pose[4] = new_quat[1]
        new_pose[5] = new_quat[2]
        new_pose[6] = new_quat[3]
        
        return new_pose
    
    
    def is_colliding(self, model1_names, model2_names):
        """
        Checks wheter one of the objects named in model1_names collides with one of the objects named in model2_names.
        
        Args:
            model1_names (list of tuples): Each tuple defines a group of objects that can collide.
            model2_names (list of tuples): Each tuple defines the group of objects that a relevant for the corresponding group in model1_names

        Returns:
            tuple: bool if a relevant collision happened,
                the contact point of the collison (x,y,z)-coordinates
                the geom_id of the first object in the collision
                the geom_id of the second object in the collision
                the geom_name of the first object in the collision
                the geom_name of the second object in the collision
        """
        
        data = self._physics.data
        
        # iterate through all contacts in the scene.
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_id, geom2_id = contact.geom1, contact.geom2
            geom1_name = self._physics.model.id2name(geom1_id, 'geom')
            geom2_name = self._physics.model.id2name(geom2_id, 'geom')
            
            for j in range(len(model1_names)):
                if (
                    (
                        geom1_name.startswith(model1_names[j]) and geom2_name.startswith(model2_names[j])
                    ) or
                    (
                        geom2_name.startswith(model1_names[j]) and geom1_name.startswith(model2_names[j])
                    )
                ):
                    contact_point = contact.pos.copy()
                    return True, contact_point, geom1_id, geom2_id, geom1_name, geom2_name

        return False, None, None, None, None, None
    
    
    def get_box_position(self, boxname):
        """
        Returns the x-, y-, z-coordinate of the given box.
        
        Args:
            model1_names (String): the name of the box of interest

        Returns:
            np.ndarray: (x,y,z) coordinates of the box position
        """
        
        box_id = self._physics.model.name2id(f"{boxname}_model/{boxname}_geom", "geom")
        return self._physics.data.geom_xpos[box_id]
    
    def get_gripper_position(self, with_quaternions=True):
        """
        Returns the x-, y-, z-coordinate and optional the quaternion of the gripper.
        
        Args:
            with_quaternion (bool): whether the quaternion should also be returned

        Returns:
            np.ndarray: (x,y,z) coordinates of the box position, optional plus quaternion values
        """
        
        if with_quaternions:
            ee_pos = self._physics.bind(self._arm.eef_site).xpos
            ee_quat = mat2quat(self._physics.bind(self._arm.eef_site).xmat.reshape(3, 3))
            ee_pose = np.concatenate([ee_pos, ee_quat])
            return ee_pose
        
        return self._physics.bind(self._arm.eef_site).xpos
    
    def get_sausage_position(self, part="middle", with_quaternions=True):
        """
        Returns the x-, y-, z-coordinate and optional the quaternion of the given part of the sausage.
        
        Args:
            part (String): part of the sausage "left", "middle", or "right"
            with_quaternion (bool): whether the quaternion should also be returned

        Returns:
            np.ndarray: (x,y,z) coordinates of the sausage part position, optional plus quaternion values
        """
        
        if with_quaternions:
            ee_pos = self._physics.bind(self._sausage.sites[f"sausage_site_{part}"]).xpos
            ee_quat = mat2quat(self._physics.bind(self._sausage.sites[f"sausage_site_{part}"]).xmat.reshape(3, 3))
            ee_pose = np.concatenate([ee_pos, ee_quat])
            return ee_pose
        
        return self._physics.bind(self._sausage.sites[f"sausage_site_{part}"]).xpos

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()