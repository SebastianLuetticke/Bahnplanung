import mujoco
import mujoco.viewer
from mujoco import mjtObj
import time
import os
import numpy as np
import math
from enum import IntEnum
import re


# Description of sensordata array values
class SENSORS_ID(IntEnum):
    # DOWN Conveyor Belt Sensors
    VELOCIMETER_X_DOWN = 0        # Velocimeter X-Value
    VELOCIMETER_Y_DOWN = 1        # Velocimeter Y-Value
    VELOCIMETER_Z_DOWN = 2        # Velocimeter Z-Value

    # UP Conveyor Belt Sensors
    VELOCIMETER_X_UP = 6          # Velocimeter X-Value
    VELOCIMETER_Y_UP = 7          # Velocimeter Y-Value
    VELOCIMETER_Z_UP = 8          # Velocimeter Z-Value


# Describes each actuator in the ctrl array.
class ACTUATORS(IntEnum):
    SHOULDER_PAN = 0
    SHOULDER_LIFT = 1
    FOREARM = 2
    WRIST_1 = 3
    WRIST_2 = 4
    WRIST_3 = 5
    GRIPPER = 6
    VELOCITY_DOWN = 7             # Roller velocity actuator for DOWN conveyor belt.
    VELOCITY_UP = 8               # Roller velocity actuator for UP conveyor belt.
    VELOCITY_FIRST_SAUSAGE = 9    # Sausage velocity actuator for the first sausage.


MODEL_PATH = "main.xml"

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Modelldatei '{MODEL_PATH}' nicht gefunden. Stelle sicher, dass sie im selben Ordner liegt.")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
pert = None
scene = mujoco.MjvScene(model, maxgeom=10000)

ROLLER_SPEED = 3 * math.pi
spawned_sausages = 0
SPAWN_INTERVAL = 2
last_spawn_time = time.time()

JOINT_START_INDEX = 12  # The startindex for the joint of the first sausage in the joint array


# Start the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        
        dt = model.opt.timestep
        
        # Set the velocities of the conveyors.
        data.ctrl[ACTUATORS.VELOCITY_DOWN] = ROLLER_SPEED
        data.ctrl[ACTUATORS.VELOCITY_UP] = -ROLLER_SPEED

        if time.time() - last_spawn_time >= SPAWN_INTERVAL:
            joint_index = JOINT_START_INDEX + spawned_sausages*7

            # Move a new sausage to the starting position of the first conveyor.
            data.qpos[joint_index : joint_index+7] = [-0.65, 1, 0.2, 0.7071, 0, -0.7071, 0]
            data.ctrl[ACTUATORS.VELOCITY_FIRST_SAUSAGE+spawned_sausages] = 0
            spawned_sausages = (spawned_sausages + 1) % 10
            last_spawn_time = time.time()
            
        
        # Supervise all collisions and handle those involving sausages and conveyors.
        for i in range(data.ncon):  # ncon = number of active contacts
            contact = data.contact[i]
            geom1 = mujoco.mj_id2name(model, mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(model, mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1 and geom2:
                # A geom name from a sausage is from the structure "index_sausage_geom".
                # Remove the prefix "index_".
                geom1_suffix = re.sub(r"^[A-Za-z0-9]+_", "", geom1)
                geom2_suffix = re.sub(r"^[A-Za-z0-9]+_", "", geom2)
                
                if "sausage_geom" in (geom1_suffix, geom2_suffix) and "belt_geom" in (geom1_suffix, geom2_suffix):
                    
                    #print(contact)
                    # Extract the index of the sausage and the belt of the collision.
                    if geom1_suffix == "sausage_geom":
                        sausage_prefix = re.match(r"^([A-Za-z0-9]+)_", geom1)
                        belt_prefix = re.match(r"^([A-Za-z0-9]+)_", geom2)
                    else:
                        sausage_prefix = re.match(r"^([A-Za-z0-9]+)_", geom2)
                        belt_prefix = re.match(r"^([A-Za-z0-9]+)_", geom1)
                    sausage_prefix = sausage_prefix.group(1) if sausage_prefix else None
                    belt_prefix = belt_prefix.group(1) if belt_prefix else None
                    
                    # Change the speed of the sausage depending on which conveyor it is lying on.
                    if belt_prefix == "down":
                        data.ctrl[ACTUATORS.VELOCITY_FIRST_SAUSAGE+int(sausage_prefix)] = data.sensordata[SENSORS_ID.VELOCIMETER_X_DOWN]
                    else:
                        data.ctrl[ACTUATORS.VELOCITY_FIRST_SAUSAGE+int(sausage_prefix)] = data.sensordata[SENSORS_ID.VELOCIMETER_X_UP]
                

        # Step simulation
        mujoco.mj_step(model, data)

        # update viewer
        viewer.sync()
        time.sleep(dt)