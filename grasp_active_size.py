import os
import sys
import time
import numpy as np
import math
import pybullet as p

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = "/home/ikeuchi/robotis"
SAC_DIR = os.path.join(WORKSPACE_DIR, "grasp/isaac/sac")
USD_PATH = os.path.join(SAC_DIR, "panda_hx5.usd")
URDF_PATH = os.path.join(SCRIPT_DIR, "panda_hx5_right.urdf")
MODEL_PATH = os.path.join(SCRIPT_DIR, "active_size.zip")

def create_inference_env(app, box_pos, box_half_extents, headless=False):
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    from omni.isaac.core.prims import RigidPrim, XFormPrim
    from omni.isaac.core.utils.types import ArticulationAction

    ARM_JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    FINGER_JOINT_NAMES = [f"finger_r_joint{i}" for i in range(1, 21)]

    class InferenceSizeEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.headless = headless
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
            
            self.previous_action = np.zeros(2, dtype=np.float32)
            self.stable_lift_count = 0
            self.is_ready_to_lift = False
            self.grasp_stable_count = 0
            self.lift_start_step = 0

            # Target position and size passed from arguments
            self.current_box_size = list(box_half_extents)
            self.current_box_pos = list(box_pos)

            self.world = World(stage_units_in_meters=1.0)
            
            # Prevent adding ground plane / table multiple times if World already has them
            if not prim_utils.is_prim_path_valid("/World/defaultGroundPlane"):
                self.world.scene.add_default_ground_plane()

            if not prim_utils.is_prim_path_valid("/World/Table"):
                self.table = FixedCuboid(
                    prim_path="/World/Table",
                    name="table",
                    position=np.array([0.6, 0.0, 0.225]),
                    scale=np.array([0.60, 0.60, 0.45]),
                    color=np.array([0.55, 0.40, 0.25]),
                )

            self.robot_prim_path = "/World/PandaHX5"
            if not prim_utils.is_prim_path_valid(self.robot_prim_path):
                if not os.path.exists(USD_PATH):
                    raise FileNotFoundError(f"USD path not found: {USD_PATH}")
                stage_utils.add_reference_to_stage(usd_path=USD_PATH, prim_path=self.robot_prim_path)
            
            self.robot = Articulation(prim_path=self.robot_prim_path, name="robot")
            self.world.scene.add(self.robot)

            self.box_prim_path = "/World/Box"
            if not prim_utils.is_prim_path_valid(self.box_prim_path):
                self.box = DynamicCuboid(
                    prim_path=self.box_prim_path,
                    name="grasp_box",
                    position=np.array(self.current_box_pos),
                    scale=np.array(self.current_box_size) * 2.0,
                    color=np.array([0.1, 0.7, 0.9]),
                    mass=0.01,
                )
                self.world.scene.add(self.box)
            else:
                self.box = RigidPrim(prim_path=self.box_prim_path, name="grasp_box")
                self.world.scene.add(self.box)

            # Set scale before initializing
            self.world.stop()
            self.box.set_local_scale(np.array(self.current_box_size) * 2.0)
            self.world.reset()
            self.world.step(render=not self.headless)
            self.robot.initialize()
            self.box.initialize()
            
            self.tcp_prim = RigidPrim(prim_path="/World/PandaHX5/tcp", name="tcp_prim")
            self.thumb_prim = XFormPrim(prim_path="/World/PandaHX5/finger_end_r_link1", name="thumb_prim")
            self.index_prim = XFormPrim(prim_path="/World/PandaHX5/finger_r_link8", name="index_prim")
            self.middle_prim = XFormPrim(prim_path="/World/PandaHX5/finger_end_r_link3", name="middle_prim")
            self.ring_prim = XFormPrim(prim_path="/World/PandaHX5/finger_r_link16", name="ring_prim")
            self.pinky_prim = XFormPrim(prim_path="/World/PandaHX5/finger_r_link20", name="pinky_prim")
            
            self.tcp_prim.initialize()
            self.thumb_prim.initialize()
            self.index_prim.initialize()
            self.middle_prim.initialize()
            self.ring_prim.initialize()
            self.pinky_prim.initialize()

            all_dofs = set(self.robot.dof_names) if self.robot.dof_names else set()
            self.arm_joint_indices = [self.robot.get_dof_index(name) for name in ARM_JOINT_NAMES if name in all_dofs]
            self.finger_joint_indices = [self.robot.get_dof_index(name) for name in FINGER_JOINT_NAMES if name in all_dofs]

            self.step_counter = 0
            self.max_steps = 50

            self.arm_grasp_q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.57, 0.785])
            self.base_finger_q = np.zeros(len(self.finger_joint_indices))
            self.current_finger_q = np.zeros(len(self.finger_joint_indices))

            # PyBullet IK setup
            self.pb_cid = p.connect(p.DIRECT)
            self.pb_robot = p.loadURDF(URDF_PATH, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True, physicsClientId=self.pb_cid)
            
            self.pb_link_indices = {}
            self.pb_joint_indices = {}
            for i in range(p.getNumJoints(self.pb_robot, physicsClientId=self.pb_cid)):
                info = p.getJointInfo(self.pb_robot, i, physicsClientId=self.pb_cid)
                j_name = info[1].decode("utf-8")
                l_name = info[12].decode("utf-8")
                self.pb_link_indices[l_name] = i
                self.pb_joint_indices[j_name] = i

            self.pb_rest_poses = [-0.01664124, 0.07485258, -0.01176343, -1.42680791, -0.01028490, 1.16893920, -2.39960431]
            for i in range(7):
                p.resetJointState(self.pb_robot, i, self.pb_rest_poses[i], physicsClientId=self.pb_cid)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            # 1. Update box scale safely
            self.world.stop()
            self.box.set_local_scale(np.array(self.current_box_size) * 2.0)
            self.world.reset()

            # 2. Update box pose
            self.box.set_world_pose(position=np.array(self.current_box_pos), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            self.box.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

            # 3. Preshape finger configuration
            dx = self.current_box_size[0]
            dy = self.current_box_size[1]
            self.base_finger_q = np.zeros(len(self.finger_joint_indices), dtype=np.float32)
            
            base_clearance = 0.30
            x_offset = max(0.0, (dx - 0.015) * 1.5)
            y_offset = (dy - 0.04) * 1.5
            
            self.base_finger_q[2::4] += (base_clearance + x_offset)
            self.base_finger_q[3::4] += (base_clearance + x_offset)
            self.base_finger_q[1::4] += y_offset * 0.5
            
            # Thumb preshape
            self.base_finger_q[0] = 0.175
            self.base_finger_q[1] = -1.745
            self.base_finger_q[2] = 0.175
            self.base_finger_q[3] = 0.0

            self.current_finger_q = np.array(self.base_finger_q)

            # 4. PyBullet IK with Iterative Sag Calibration
            for i in range(7):
                p.resetJointState(self.pb_robot, self.pb_joint_indices[f"panda_joint{i+1}"], self.pb_rest_poses[i], physicsClientId=self.pb_cid)
            for i in range(20):
                j_name = f"finger_r_joint{i+1}"
                if j_name in self.pb_joint_indices:
                    p.resetJointState(self.pb_robot, self.pb_joint_indices[j_name], self.base_finger_q[i], physicsClientId=self.pb_cid)

            pb_tcp_state = p.getLinkState(self.pb_robot, self.pb_link_indices["tcp"], physicsClientId=self.pb_cid)
            base_tcp_pos = np.array(pb_tcp_state[0])
            base_tcp_quat = np.array(pb_tcp_state[1])
            
            pb_thumb = np.array(p.getLinkState(self.pb_robot, self.pb_link_indices["finger_end_r_link1"], physicsClientId=self.pb_cid)[0])
            pb_middle = np.array(p.getLinkState(self.pb_robot, self.pb_link_indices["finger_end_r_link3"], physicsClientId=self.pb_cid)[0])
            base_midpoint = (pb_thumb + pb_middle) / 2.0
            base_midpoint_offset = base_midpoint - base_tcp_pos
            
            box_center_pos = np.array([self.current_box_pos[0], self.current_box_pos[1], self.current_box_pos[2]])
            z_offset = 0.015
            target_midpoint = box_center_pos + np.array([0.0, 0.0, z_offset])
            pb_target_midpoint = np.copy(target_midpoint)
            
            self.arm_grasp_q = np.zeros(7, dtype=np.float32)
            
            for it in range(5):
                pb_target_tcp = pb_target_midpoint - base_midpoint_offset
                
                q_ik = p.calculateInverseKinematics(
                    self.pb_robot,
                    self.pb_link_indices["tcp"],
                    pb_target_tcp,
                    targetOrientation=base_tcp_quat,
                    maxNumIterations=200,
                    residualThreshold=1e-5,
                    physicsClientId=self.pb_cid
                )
                
                for i in range(7):
                    self.arm_grasp_q[i] = q_ik[self.pb_joint_indices[f"panda_joint{i+1}"]]
                    
                self.robot.set_joint_positions(self.arm_grasp_q, self.arm_joint_indices)
                self.robot.set_joint_positions(self.base_finger_q, self.finger_joint_indices)
                
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.arm_grasp_q, joint_indices=self.arm_joint_indices)
                )
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.base_finger_q, joint_indices=self.finger_joint_indices)
                )
                
                for _ in range(10):
                    self.box.set_world_pose(position=self.current_box_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
                    self.box.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                    self.box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    self.world.step(render=False)
                
                isaac_thumb, _ = self.thumb_prim.get_world_pose()
                isaac_middle, _ = self.middle_prim.get_world_pose()
                current_midpoint = (isaac_thumb + isaac_middle) / 2.0
                
                error = target_midpoint - current_midpoint
                err_norm = np.linalg.norm(error)
                
                if err_norm < 0.002:
                    break
                    
                pb_target_midpoint = pb_target_midpoint + error * 0.8

            self.robot.set_joint_positions(self.arm_grasp_q, self.arm_joint_indices)
            self.robot.set_joint_positions(self.base_finger_q, self.finger_joint_indices)

            self.robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.arm_grasp_q, joint_indices=self.arm_joint_indices)
            )
            self.robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.base_finger_q, joint_indices=self.finger_joint_indices)
            )

            for _ in range(10):
                self.world.step(render=not self.headless)

            self.step_counter = 0
            self.stable_lift_count = 0
            self.is_ready_to_lift = False
            self.grasp_stable_count = 0
            self.lift_start_step = 0
            self.previous_action = np.zeros(2, dtype=np.float32)
            
            return self._get_obs(), {}

        def _get_obs(self):
            self.world.step(render=False)
            obs = []
            robot_dof_pos = self.robot.get_joint_positions()
            obs.extend(robot_dof_pos[:15] if len(robot_dof_pos) >= 15 else np.pad(robot_dof_pos, (0, 15 - len(robot_dof_pos))))
            box_pos, _ = self.box.get_world_pose()
            obs.extend(box_pos)
            finger_pos = robot_dof_pos[7:27] if len(robot_dof_pos) >= 27 else np.pad(robot_dof_pos[7:], (0, 20 - len(robot_dof_pos[7:])))
            obs.extend(finger_pos)
            obs.extend(self.current_box_size)
            measured_efforts = self.robot.get_measured_joint_efforts()
            finger_efforts = measured_efforts[7:27] if len(measured_efforts) >= 27 else np.pad(measured_efforts[7:], (0, 20 - len(measured_efforts[7:])))
            obs.extend(finger_efforts)
            obs.extend(self.previous_action)
            step_progress = self.step_counter / self.max_steps
            obs.append(step_progress)
            return np.array(obs[:64], dtype=np.float32)

        def step(self, action):
            self.step_counter += 1
            action_clipped = np.clip(action, -1.0, 1.0)
            self.previous_action = np.copy(action_clipped)
            a_thumb = action_clipped[0]
            a_finger = action_clipped[1]

            delta_thumb = a_thumb * 0.02
            delta_finger = a_finger * 0.03

            for i in range(min(5, len(self.finger_joint_indices) // 4)):
                val = delta_thumb if i == 0 else delta_finger
                idx = i * 4
                if idx + 3 < len(self.current_finger_q):
                    if i == 0:
                        self.current_finger_q[idx + 2] = np.clip(self.current_finger_q[idx + 2] + val, -0.2, 0.65)
                        self.current_finger_q[idx + 3] = np.clip(self.current_finger_q[idx + 3] + val, -0.2, 0.65)
                    else:
                        self.current_finger_q[idx + 1] = np.clip(self.current_finger_q[idx + 1] + val, -0.2, 0.65)
                        self.current_finger_q[idx + 2] = np.clip(self.current_finger_q[idx + 2] + val, -0.2, 0.65)
                        self.current_finger_q[idx + 3] = np.clip(self.current_finger_q[idx + 3] + val, -0.2, 0.65)

            if len(self.finger_joint_indices) > 0:
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.current_finger_q, joint_indices=self.finger_joint_indices)
                )

            for _ in range(5):
                self.world.step(render=not self.headless)

            box_pos, _ = self.box.get_world_pose()
            lift_height = box_pos[2] - self.current_box_pos[2]
            
            thumb_pos, _ = self.thumb_prim.get_world_pose()
            index_pos, _ = self.index_prim.get_world_pose()
            middle_pos, _ = self.middle_prim.get_world_pose()
            ring_pos, _ = self.ring_prim.get_world_pose()
            pinky_pos, _ = self.pinky_prim.get_world_pose()
            
            dist_thumb = np.linalg.norm(thumb_pos - box_pos)
            dist_index = np.linalg.norm(index_pos - box_pos)
            dist_middle = np.linalg.norm(middle_pos - box_pos)
            dist_ring = np.linalg.norm(ring_pos - box_pos)
            dist_pinky = np.linalg.norm(pinky_pos - box_pos)
            avg_finger_dist = (dist_thumb + dist_index + dist_middle + dist_ring + dist_pinky) / 5.0
            
            measured_efforts = self.robot.get_measured_joint_efforts()
            finger_efforts = measured_efforts[7:27] if len(measured_efforts) >= 27 else np.zeros(20)
            total_finger_effort = np.sum(np.abs(finger_efforts))

            is_force_balanced = (total_finger_effort > 0.45) and (avg_finger_dist < 0.09) and (self.step_counter > 15)

            if is_force_balanced and not self.is_ready_to_lift:
                self.grasp_stable_count += 1
            elif not self.is_ready_to_lift:
                self.grasp_stable_count = 0

            if self.grasp_stable_count >= 5 and not self.is_ready_to_lift:
                self.is_ready_to_lift = True
                self.lift_start_step = self.step_counter

            if self.is_ready_to_lift:
                target_lift_q = np.array(self.arm_grasp_q)
                lift_progress = min(20.0, self.step_counter - self.lift_start_step)
                target_lift_q[1] -= 0.3 * (lift_progress / 20.0)
                self.robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=target_lift_q, joint_indices=self.arm_joint_indices)
                )

            done = False
            info = {}

            if self.is_ready_to_lift:
                if lift_height > 0.025:
                    self.stable_lift_count += 1
                else:
                    self.stable_lift_count = 0

                if self.stable_lift_count >= 10:
                    info["is_success"] = True
                    done = True

            if self.step_counter >= self.max_steps and not done:
                done = True
                info["is_success"] = False

            obs = self._get_obs()
            return obs, 0.0, done, False, info

        def update_box_params(self, box_pos, box_half_extents):
            self.current_box_pos = list(box_pos)
            self.current_box_size = list(box_half_extents)

        def close(self):
            try:
                p.disconnect(self.pb_cid)
            except Exception:
                pass

    return InferenceSizeEnv()


# Global singleton cache
_active_app = None
_active_env = None
_active_model = None

def grasp_active_size(x, y, z, hx, hy, hz, app=None, headless=False):
    """
    Executes the trained grasping policy for a box at (x,y,z) with half-extents (hx,hy,hz).
    Uses the latest size_cal2 SAC model (checkpoints2) with PyBullet Iterative IK calibration.

    Args:
        x, y, z (float): Center position of the box in world coordinates [meters]
        hx, hy, hz (float): Half-extents of the box [meters] (e.g. 0.025 = 5cm width)
        app: Optional existing Isaac Sim SimulationApp instance
        headless (bool): If True, runs without GUI rendering

    Returns:
        bool: True if grasping and lifting succeeded, False otherwise
    """
    global _active_app, _active_env, _active_model

    box_pos = [x, y, z]
    box_half_extents = [hx, hy, hz]

    # 1. Initialize SimulationApp if needed
    if app is None and _active_app is None:
        print("[grasp_active_size] Starting Isaac Sim App...")
        from omni.isaac.kit import SimulationApp
        _active_app = SimulationApp({"headless": headless})
    elif app is not None:
        _active_app = app

    # 2. Create or reuse Environment
    if _active_env is None:
        print("[grasp_active_size] Creating Inference Environment...")
        _active_env = create_inference_env(_active_app, box_pos, box_half_extents, headless=headless)
        
        if not headless:
            import numpy as np
            from omni.isaac.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([1.0, 0.5, 0.7]), target=np.array([0.6, 0.0, 0.4]))
    else:
        _active_env.update_box_params(box_pos, box_half_extents)

    # 3. Load SAC Model
    if _active_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        print(f"[grasp_active_size] Loading SAC model from {MODEL_PATH}...")
        vec_env = DummyVecEnv([lambda: _active_env])
        _active_model = SAC.load(MODEL_PATH, env=vec_env, device="cuda:0")

    print(f"[grasp_active_size] Executing grasp for box at {box_pos} with size {box_half_extents}...")
    
    obs, _ = _active_env.reset()
    if not headless:
        import numpy as np
        from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([0.88, 0.22, 0.58]), target=np.array([0.635, 0.0, 0.475]))
    
    success = False
    for step in range(50):
        action, _ = _active_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = _active_env.step(action)
        
        if not headless:
            time.sleep(0.02)
        
        if done:
            success = info.get("is_success", False)
            break

    print(f"[grasp_active_size] Grasping finished. Result: {'SUCCESS' if success else 'FAILED'}")
    return success
