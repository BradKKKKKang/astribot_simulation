#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import json
import re
import threading
import numpy as np
from typing import Optional, Dict, List, TYPE_CHECKING

os.environ["CARB_LOGGING_ENABLED"] = "0"
os.environ["OMNI_LOG_LEVEL"] = "error"
os.environ["ISAAC_VERBOSE"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ["omni", "isaac", "carb", "omni.kit", "omni.isaac",
                    "isaacsim", "omni.physx", "omni.replicator", "omni.ext",
                    "omni.usd", "omni.hydra", "omni.graph"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

from isaaclab.app import AppLauncher

from src.simu_utils.simu_common_tools import astribot_simu_log
from src.astribot_envs.astribot_base_env import AstribotBaseEnv
from src.simulation_config.isaac_scene_config import (
    CAMERA_BODIES, SUPPORTED_ROBOT_ROOT, camera_prim_path,
    normalize_scene_simulation, spawn_scene_environment,
)

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


class IsaacLabAppManager:
    _instance = None
    _app_launcher = None
    _simulation_app = None
    _ref_count = 0
    _modules_loaded = False

    @staticmethod
    def _suppress_output():
        class SuppressOutput:
            def __enter__(self):
                sys.stdout.flush()
                sys.stderr.flush()
                self.original_stdout_fd = sys.stdout.fileno()
                self.original_stderr_fd = sys.stderr.fileno()

                self.saved_stdout_fd = os.dup(self.original_stdout_fd)
                self.saved_stderr_fd = os.dup(self.original_stderr_fd)

                self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

                os.dup2(self.devnull_fd, self.original_stdout_fd)
                os.dup2(self.devnull_fd, self.original_stderr_fd)

                return self

            def __exit__(self, *args):
                # Kit retains Python stream references during startup. Redirect only
                # their file descriptors; closing replacement streams breaks later logs.
                sys.stdout.flush()
                sys.stderr.flush()

                os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
                os.dup2(self.saved_stderr_fd, self.original_stderr_fd)

                os.close(self.saved_stdout_fd)
                os.close(self.saved_stderr_fd)
                os.close(self.devnull_fd)

        return SuppressOutput()

    @classmethod
    def get_or_create_app(
        cls, headless: bool = False, width: int = 1920, height: int = 1080,
        enable_cameras: bool = False,
    ):
        if cls._instance is None:
            cls._instance = cls()

        if cls._app_launcher is None:
            if enable_cameras:
                # Load Pillow's Python and binary modules together before Kit
                # adds its bundled Pillow version to the import paths.
                from PIL import Image  # noqa: F401

            with cls._suppress_output():
                import argparse
                parser = argparse.ArgumentParser()
                AppLauncher.add_app_launcher_args(parser)

                args_list = []
                if headless:
                    args_list.append("--headless")
                if enable_cameras:
                    args_list.append("--enable_cameras")

                args = parser.parse_args(args_list)
                # Isaac Sim 4.5: use synchronized rendering and let the Hydra
                # scene delegate consume PhysX's Fabric transforms.
                args.kit_args = (
                    "--/app/asyncRendering=false "
                    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false "
                    "--/persistent/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"
                )
                cls._app_launcher = AppLauncher(args)
                cls._simulation_app = cls._app_launcher.app

            try:
                import carb
                settings = carb.settings.get_settings()

                if not headless:
                    settings.set("/app/window/width", width)
                    settings.set("/app/window/height", height)

                settings.set("/app/log/outputStreamLevel", 40)
                settings.set("/app/extensions/verboseLogging", False)
            except Exception:
                pass

            cls._load_isaaclab_modules()

        cls._ref_count += 1
        return cls._app_launcher, cls._simulation_app

    @classmethod
    def _load_isaaclab_modules(cls):
        if not cls._modules_loaded:
            global torch, sim_utils, Articulation, ArticulationCfg
            global InteractiveScene, InteractiveSceneCfg, SimulationCfg, SimulationContext

            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation, ArticulationCfg
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            from isaaclab.sim import SimulationCfg, SimulationContext

            cls._modules_loaded = True

    @classmethod
    def release_app(cls):
        if cls._instance is not None:
            cls._ref_count -= 1
            if cls._ref_count <= 0:
                if cls._simulation_app is not None:
                    astribot_simu_log("Closing Isaac Lab simulation app...")
                    cls._simulation_app.close()
                cls._app_launcher = None
                cls._simulation_app = None
                cls._modules_loaded = False
                cls._ref_count = 0


class AstribotIsaacLabEnv(AstribotBaseEnv):

    # ROS/dataset gripper commands use 0..100; Isaac joint targets use radians.
    _GRIPPER_RAD_PER_PERCENT = 0.93 / 100.0
    _GRIPPER_JOINT_DIRECTIONS = {
        'L1': -1, 'L2': -1, 'R11': -1,
        'R1': 1, 'R2': 1, 'L11': 1,
    }
    _CONTROL_PATTERNS = {
        **{f'torso_joint_{i}_control': f'astribot_torso_joint_{i}' for i in (1, 2, 3)},
        **{f'gripper_{side}_control': f'astribot_gripper_{side}_joint_(L1|L2|L11|R1|R2|R11)'
           for side in ('left', 'right')},
    }

    def __init__(self, param):
        astribot_simu_log("Initializing Isaac Lab environment...")
        self.scene_simulation = normalize_scene_simulation(param.get('scene_simulation'))
        self.scene_object_specs = param.get('scene_objects', [])
        self.task_reset = param.get('task_reset')
        self.task_scene = param.get('task_scene')
        self.scene_object_keys = {'dynamic': {}, 'static': {}, 'resources': None}
        self.robot_placement_declaration = param.get('robot_placement')
        self.robot_profile_data = param.get('robot_profile')
        self.resolved_robot_placement = None
        self._placement_publisher = None
        self.robot_root_path = param.get('robot_root_path', SUPPORTED_ROBOT_ROOT)
        if self.robot_root_path != SUPPORTED_ROBOT_ROOT:
            raise ValueError('Isaac scene adapter requires /World/astribot_whole_body robot root')
        self.camera_bodies = {**CAMERA_BODIES, **param.get('camera_bodies', {})}
        for camera_name in param.get('camera_names', []):
            camera_prim_path(camera_name, self.camera_bodies.get(camera_name, ''), self.robot_root_path)

        super().__init__(param)

        self.dt = param.get('dt', 0.01)
        if self.managed_evaluation and self.dt != 0.01:
            raise ValueError('managed evaluation requires dt=0.01')
        self.decimation = param.get('decimation', 1)
        self.physics_device = param.get("physics_device")
        self.use_fabric = param.get("use_fabric", True)
        for name in self._CONTROL_PATTERNS:
            setattr(self, name, self._validate_joint_control(name, param.get(name, {})))
        self.camera_resolutions = {
            name: (
                param.get("camera_resolutions", {}).get(name, {}).get("width", self.width),
                param.get("camera_resolutions", {}).get(name, {}).get("height", self.height),
            )
            for name in self.camera_names
        }
        self._camera_output_logged = False
        self.third_person_camera_cfg = param.get("third_person_camera")
        self._video_lock = threading.Lock()
        self._video_start_requested = False
        self._video_stop_requested = False
        self._video_started = threading.Event()
        self._video_stopped = threading.Event()
        self._video_writer = None
        self._video_error = None
        self._video_frame_count = 0
        self._video_start_sim_time = None
        self._video_end_sim_time = None
        self._video_next_capture_time = None
        self._video_record_service = None
        if self.third_person_camera_cfg:
            from std_srvs.srv import SetBool

            self._video_record_service = self.node.create_service(
                SetBool, "/simulation/video_recording", self._handle_video_recording,
            )

        headless = (self.render_mode != "human")
        self.app_launcher, self.simulation_app = IsaacLabAppManager.get_or_create_app(
            headless=headless,
            width=self.width,
            height=self.height,
            enable_cameras=bool(self.camera_names or self.third_person_camera_cfg),
        )

        self.setup_simulation_and_scene()

        self.setup_joint_mapping()
        if self.resolved_robot_placement is not None and self.node is not None:
            from std_msgs.msg import String
            self._placement_publisher = self.node.create_publisher(String, '/simulation/robot_placement', 10)
            self._publish_robot_placement()

        self.joint_names_all, self.controller_mode, self.joint_position_command_all, \
            self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()

        astribot_simu_log("Isaac Lab environment initialized successfully")

    def setup_simulation_and_scene(self):
        import torch
        self.device = self.physics_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.robot_placement_declaration is not None:
            from src.simulation_config.robot_profile import RobotProfile
            from src.simulation_config.robot_placement import resolve_robot_placement
            if self.robot_profile_data is None:
                raise ValueError('explicit robot placement requires a shared robot profile')
            self.resolved_robot_placement = resolve_robot_placement(
                RobotProfile(self.robot_profile_data), self.robot_placement_declaration)

        sim_cfg = SimulationCfg(
            dt=self.dt,
            render_interval=self.decimation,
            device=self.device,
            use_fabric=self.use_fabric,
            gravity=(0.0, 0.0, -9.81),
            physics_prim_path="/physicsScene",
        )

        self.sim = SimulationContext(sim_cfg)

        astribot_simu_log(f"Physics device: {self.device}, Fabric: {self.use_fabric}")

        spawn_scene_environment(sim_utils, self.scene_simulation)

        scene_cfg = InteractiveSceneCfg(
            num_envs=1,
            env_spacing=3.0,
        )

        robot_cfg = self._create_robot_config()
        scene_cfg.robot = robot_cfg
        from src.simulation_config.isaac_scene_objects import attach_scene_objects
        self.scene_object_keys = attach_scene_objects(scene_cfg, self.scene_object_specs)

        if self.camera_names:
            if not self.model_path.endswith(('.usd', '.usda')):
                raise ValueError("Isaac Lab cameras require the USD robot model")
            from isaaclab.sensors import CameraCfg

            for camera_name in self.camera_names:
                if camera_name not in self.camera_bodies:
                    raise ValueError(f"Unknown Isaac Lab camera: {camera_name}")
                width, height = self.camera_resolutions[camera_name]
                # Existing USD cameras already use the OpenGL convention.
                # spawn=None preserves their link-relative poses and optics.
                setattr(scene_cfg, camera_name, CameraCfg(
                    prim_path=camera_prim_path(camera_name, self.camera_bodies[camera_name],
                                               self.robot_root_path),
                    spawn=None,
                    data_types=["rgb"],
                    width=width,
                    height=height,
                ))

        if self.third_person_camera_cfg:
            if not self.model_path.endswith((".usd", ".usda")):
                raise ValueError("Third-person recording requires the USD robot model")
            from isaaclab.sensors import CameraCfg

            width = int(self.third_person_camera_cfg.get("width", 1280))
            height = int(self.third_person_camera_cfg.get("height", 720))
            fps = float(self.third_person_camera_cfg.get("fps", 30.0))
            eye = tuple(float(value) for value in self.third_person_camera_cfg["eye"])
            if len(eye) != 3 or width <= 0 or height <= 0 or fps <= 0:
                raise ValueError("Invalid third_person_camera dimensions, fps, or eye")
            setattr(scene_cfg, "third_person_camera", CameraCfg(
                prim_path="{ENV_REGEX_NS}/third_person_camera",
                offset=CameraCfg.OffsetCfg(pos=eye, convention="world"),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(self.third_person_camera_cfg.get("focal_length", 24.0)),
                    focus_distance=10.0,
                    clipping_range=(0.1, 30.0),
                ),
                data_types=["rgb"],
                width=width,
                height=height,
                update_period=1.0 / fps,
            ))

        self.scene = InteractiveScene(scene_cfg)

        self.robot: Articulation = self.scene["robot"]

        if self.resolved_robot_placement is not None:
            from src.simulation_config.robot_placement import transform_world_fixed_anchors
            import omni.usd
            transform_world_fixed_anchors(omni.usd.get_context().get_stage(),
                '/World/envs/env_0/Robot/astribot_whole_body',
                self.resolved_robot_placement['world_transform'])

        self.sim.reset()
        self._restore_initial_state()
        if self.third_person_camera_cfg:
            eye = torch.tensor(
                [self.third_person_camera_cfg["eye"]], device=self.device, dtype=torch.float32,
            )
            target = torch.tensor(
                [self.third_person_camera_cfg.get("target", (0.0, 0.0, 0.9))],
                device=self.device,
                dtype=torch.float32,
            )
            self.scene["third_person_camera"].set_world_poses_from_view(eye, target)

    def _handle_video_recording(self, request, response):
        if not self.third_person_camera_cfg:
            response.success = False
            response.message = "Third-person camera is not configured"
            return response

        timeout_s = float(self.third_person_camera_cfg.get("control_timeout_s", 10.0))
        if request.data:
            with self._video_lock:
                if self._video_writer is not None or self._video_start_requested:
                    response.success = False
                    response.message = "Video recording is already active"
                    return response
                self._video_error = None
                self._video_started.clear()
                self._video_stopped.clear()
                self._video_start_requested = True
            if not self._video_started.wait(timeout_s):
                response.success = False
                response.message = "Timed out waiting for the first recorded frame"
                return response
            if self._video_error:
                response.success = False
                response.message = self._video_error
                return response
            response.success = True
            response.message = json.dumps(self._video_metadata(), allow_nan=False)
            return response

        with self._video_lock:
            if self._video_writer is None and not self._video_start_requested:
                response.success = False
                response.message = "Video recording is not active"
                return response
            self._video_stop_requested = True
        if not self._video_stopped.wait(timeout_s):
            response.success = False
            response.message = "Timed out waiting for video encoder shutdown"
            return response
        if self._video_error:
            response.success = False
            response.message = self._video_error
            return response
        response.success = True
        response.message = json.dumps(self._video_metadata(), allow_nan=False)
        return response

    def _video_metadata(self):
        fps = float(self.third_person_camera_cfg["fps"])
        return {
            "path": str(self.third_person_camera_cfg["output_path"]),
            "fps": fps,
            "width": int(self.third_person_camera_cfg["width"]),
            "height": int(self.third_person_camera_cfg["height"]),
            "frame_count": self._video_frame_count,
            "sim_time_start_s": self._video_start_sim_time,
            "sim_time_end_s": self._video_end_sim_time,
            "sim_duration_s": (
                self._video_end_sim_time - self._video_start_sim_time
                if self._video_start_sim_time is not None and self._video_end_sim_time is not None
                else 0.0
            ),
        }

    def _capture_video_frame(self):
        if not self.third_person_camera_cfg:
            return

        import cv2

        current_sim_time = float(self.sim.current_time)
        with self._video_lock:
            if self._video_start_requested and self._video_writer is None:
                output_path = os.path.abspath(self.third_person_camera_cfg["output_path"])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fps = float(self.third_person_camera_cfg["fps"])
                size = (
                    int(self.third_person_camera_cfg["width"]),
                    int(self.third_person_camera_cfg["height"]),
                )
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    size,
                )
                if not writer.isOpened():
                    self._video_error = f"Could not open MP4 writer: {output_path}"
                    self._video_start_requested = False
                    self._video_started.set()
                    self._video_stopped.set()
                    return
                self._video_writer = writer
                self._video_frame_count = 0
                self._video_start_sim_time = current_sim_time
                self._video_end_sim_time = current_sim_time
                self._video_next_capture_time = current_sim_time
                self._video_start_requested = False

            if self._video_stop_requested and self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
                self._video_stop_requested = False
                self._video_stopped.set()
                return

            writer = self._video_writer
            next_capture = self._video_next_capture_time
            if writer is None or next_capture is None or current_sim_time + 1e-8 < next_capture:
                return

            rgb = self.scene["third_person_camera"].data.output["rgb"][0]
            frame = np.ascontiguousarray(rgb.detach().cpu().numpy())
            if frame.shape != (int(self.third_person_camera_cfg["height"]),
                               int(self.third_person_camera_cfg["width"]), 3):
                self._video_error = f"Unexpected third-person frame shape: {frame.shape}"
                writer.release()
                self._video_writer = None
                self._video_stopped.set()
                self._video_started.set()
                return
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._video_frame_count += 1
            self._video_end_sim_time = current_sim_time
            period = 1.0 / float(self.third_person_camera_cfg["fps"])
            while self._video_next_capture_time <= current_sim_time + 1e-8:
                self._video_next_capture_time += period
            if self._video_frame_count == 1:
                self._video_started.set()

    def _create_robot_config(self) -> ArticulationCfg:
        is_usd = self.model_path.endswith(('.usd', '.usda'))

        if is_usd:
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=self.model_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            )
        else:
            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=self.model_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            )

        from isaaclab.actuators import ImplicitActuatorCfg

        # Disjoint actuator groups allow isolated torso and gripper tuning.
        excluded = '|'.join(self._CONTROL_PATTERNS.values())
        actuators_cfg = {
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[f"^(?!(?:{excluded})$).*$"],
                stiffness=800.0,
                damping=40.0,
                effort_limit_sim=400.0,
            ),
        }
        for name, pattern in self._CONTROL_PATTERNS.items():
            settings = getattr(self, name)
            actuators_cfg[name.replace('_control', '_override')] = ImplicitActuatorCfg(
                joint_names_expr=[f'^{pattern}$'],
                stiffness=settings['stiffness'], damping=settings['damping'],
                effort_limit_sim=settings['effort_limit'],
                # In Isaac Lab 2.1, velocity_limit is ignored by implicit drives.
                # None preserves authored USD limits for angular joints.
                velocity_limit_sim=settings.get('velocity_limit_sim'),
            )

        placement = self.resolved_robot_placement
        spawn_pose = placement['spawn_pose_world'] if placement else [0., 0., 0., 1., 0., 0., 0.]
        initial_joints = placement['initial_joint_positions'] if placement else {'.*': 0.0}
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(spawn_pose[:3]),
                rot=tuple(spawn_pose[3:]),
                joint_pos=initial_joints,
                joint_vel={".*": 0.0},
            ),
            actuators=actuators_cfg,
        )

        return robot_cfg

    @classmethod
    def _validate_joint_control(cls, name, settings):
        if name not in cls._CONTROL_PATTERNS:
            raise ValueError(f'Unknown joint control group: {name}')
        defaults = dict(stiffness=800.0, damping=40.0, effort_limit=400.0)
        if name.startswith('torso'):
            defaults['gravity_compensation_scale'] = 0.0
        else:
            defaults['velocity_limit_sim'] = 1.0  # Historical USD gripper limit, rad/s.
        if not isinstance(settings, dict) or set(settings) - defaults.keys():
            raise ValueError(f"Unknown {name} settings")
        values = {key: float(settings.get(key, value)) for key, value in defaults.items()}
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"{name} values must be finite and non-negative")
        if any(values.get(key) == 0 for key in ('effort_limit', 'velocity_limit_sim')):
            raise ValueError(f"{name} effort and velocity limits must be positive")
        return values

    @classmethod
    def _validate_torso_joint_1_control(cls, settings):
        return cls._validate_joint_control('torso_joint_1_control', settings)

    def configure_torso_joint_1(self, settings):
        self.configure_joint_control('torso_joint_1_control', settings)

    def configure_joint_control(self, name, settings):
        """Update one actuator group between replay trials on the sim thread."""
        import torch

        values = self._validate_joint_control(name, settings)
        ids = [idx for joint, idx in self.joint_name_to_index.items()
               if re.fullmatch(self._CONTROL_PATTERNS[name], joint)]
        actuator = self.robot.actuators[name.replace('_control', '_override')]
        for key, writer in (("stiffness", self.robot.write_joint_stiffness_to_sim),
                            ("damping", self.robot.write_joint_damping_to_sim),
                            ("effort_limit", self.robot.write_joint_effort_limit_to_sim)):
            getattr(actuator, key).fill_(values[key])
            writer(torch.full((1, len(ids)), values[key], device=self.device), joint_ids=ids)
        actuator.effort_limit_sim.fill_(values["effort_limit"])
        if 'velocity_limit_sim' in values:
            limit = values['velocity_limit_sim']
            actuator.velocity_limit.fill_(limit)
            actuator.velocity_limit_sim.fill_(limit)
            self.robot.write_joint_velocity_limit_to_sim(
                torch.full((1, len(ids)), limit, device=self.device), joint_ids=ids)
        setattr(self, name, values)
        self.robot.set_joint_effort_target(torch.zeros((1, len(ids)), device=self.device), joint_ids=ids)

    def setup_joint_mapping(self):
        astribot_simu_log("Setting up joint mapping...")

        self.joint_name_to_index = {}
        self.joint_index_to_name = {}

        all_joint_names = self.robot.data.joint_names

        for idx, joint_name in enumerate(all_joint_names):
            self.joint_name_to_index[joint_name] = idx
            self.joint_index_to_name[idx] = joint_name

    def step(self, action: np.ndarray) -> tuple:
        self.update_reset_flag()

        if not self.reset_flag:
            step_begin_time = time.time()

            self.joint_names_all, self.controller_mode, self.joint_position_command_all, \
                self.joint_velocity_command_all, self.joint_torque_command_all = self.update_joint_states()

            self.update_object_states()
            self.update_trajectory_pose()
            self.update_com_pose()
            self.update_sensor_states()

            self._apply_control()

            for _ in range(self.frame_skip):
                self.scene.write_data_to_sim()
                self.sim.step(render=True)
                self.scene.update(dt=self.dt)

            self._capture_video_frame()
            self.render()
            self._publish_robot_placement()

            step_end_time = time.time()
            self.real_time_fps = 1 / (step_end_time - step_begin_time) if (step_end_time - step_begin_time) > 0 else 50
        else:
            self.reset()

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def set_managed_lease(self, enabled):
        # Fence legacy callbacks while transitioning ownership. Replay starts
        # before its SDK is constructed, so its first free step must hold the
        # measured boundary rather than an old Eval/reset command buffer.
        self.multi_robot_ros_interface.managed_evaluation = True
        for interface in self.robot_dict.values():
            interface.managed_evaluation = True
        self.multi_robot_ros_interface.reset_flag = False
        self.multi_robot_ros_interface.pose_tuple_list.clear()
        if not enabled:
            for part, interface in self.robot_dict.items():
                values = list(self.get_joint_positions(self.robot_joint_map[part]))
                interface.controller_mode = 1
                interface.joint_position_command = values
                interface.joint_velocity_command = [0.] * len(values)
                interface.joint_torque_command = [0.] * len(values)
        self.managed_evaluation = bool(enabled)
        self.multi_robot_ros_interface.managed_evaluation = bool(enabled)
        for interface in self.robot_dict.values():
            interface.managed_evaluation = bool(enabled)

    def managed_step(self, commands):
        """Execute one validated whole-body batch on the Isaac main thread."""
        if not self.managed_evaluation:
            raise RuntimeError('managed stepping is disabled')
        if not self.sim.is_playing():
            raise RuntimeError('Isaac timeline is paused or stopped; managed step cannot execute')
        names, positions, velocities = [], [], []
        for part, command in commands.items():
            names.extend(self.robot_joint_map[part])
            positions.extend(command['position'])
            velocities.extend(command['velocity'])
        self.joint_names_all = names
        self.controller_mode = [1] * len(names)
        self.joint_position_command_all = positions
        self.joint_velocity_command_all = velocities
        self.joint_torque_command_all = [0.] * len(names)
        self._apply_control()
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.dt)
        self.managed_publish_feedback()

    def managed_publish_feedback(self):
        """Publish measured joints only after a completed reset or physics step."""
        for part, names in self.robot_joint_map.items():
            self.robot_dict[part].publish_joint_states(
                self.get_joint_positions(names), self.get_joint_velocities(names),
                self.get_joint_accelerations(names), self.get_joint_torques(names))
        self._publish_robot_placement()

    def managed_snapshot(self, *, capture_images):
        """Capture cameras and joints at the same paused physics boundary."""
        import base64
        import cv2
        joints = {}
        for part in self.robot_profile_data['parts']:
            name = part['name']
            if name not in self.robot_joint_map:
                continue
            model = self.robot_joint_map[name]
            joints[name] = {'joint_names': part['feedback_joints'],
                            'position': list(self.get_joint_positions(model)),
                            'velocity': list(self.get_joint_velocities(model))}
        joints['astribot_chassis'] = {
            'joint_names': ['astribot_chassis_x', 'astribot_chassis_y', 'astribot_chassis_z_rot'],
            'position': [0., 0., 0.], 'velocity': [0., 0., 0.]}
        images = {}
        if capture_images:
            # Rendering is allowed while physics is paused. Force sensor refresh,
            # including reset at the same simulation timestamp as the prior frame.
            self.sim.render()
            for camera in self.camera_names:
                self.scene[camera].update(0.0, force_recompute=True)
                rgb = self.render_single_camera(camera)['rgb_img'][:, :, :3]
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError('snapshot JPEG encoding failed: ' + camera)
                images[camera] = {'encoding': 'jpeg', 'data_base64': base64.b64encode(encoded).decode('ascii'),
                                  'width': int(rgb.shape[1]), 'height': int(rgb.shape[0])}
        return {'joints': joints, 'images': images}

    def managed_idle(self):
        """Pump Kit/GUI rendering without advancing the physics simulation."""
        self.sim.render()

    def _apply_control(self):
        import torch

        joint_position_targets = torch.zeros(self.robot.num_joints, device=self.device)
        joint_effort_targets = torch.zeros(self.robot.num_joints, device=self.device)

        for joint_id, mode in enumerate(self.controller_mode):
            if joint_id >= len(self.joint_names_all):
                continue

            joint_name = self.joint_names_all[joint_id]
            if joint_name not in self.joint_name_to_index:
                continue

            robot_joint_idx = self.joint_name_to_index[joint_name]

            if mode == 1:
                pos_cmd = self.vel_compensation_ctrl(
                    self.joint_position_command_all,
                    self.joint_velocity_command_all,
                    idx=joint_id
                )[0]
                joint_position_targets[robot_joint_idx] = pos_cmd

            elif mode == 2:
                current_pos = self.get_joint_positions([joint_name])[0]
                vel_cmd = self.vel_compensation_ctrl(
                    [current_pos],
                    [self.joint_velocity_command_all[joint_id]],
                    idx=0
                )[0]
                joint_position_targets[robot_joint_idx] = vel_cmd

            elif mode == 3:
                joint_effort_targets[robot_joint_idx] = self.joint_torque_command_all[joint_id]

        self._sync_gripper_joints(joint_position_targets)

        # Each torso DOF has independent gravity feedforward. Fetch the current
        # generalized forces once, and never compensate unselected joints.
        self.robot.set_joint_position_target(joint_position_targets.unsqueeze(0))
        forces = None
        for i in (1, 2, 3):
            torso_idx = self.joint_name_to_index.get(f'astribot_torso_joint_{i}')
            if torso_idx is None:
                continue
            scale = getattr(self, f'torso_joint_{i}_control', {}).get('gravity_compensation_scale', 0.0)
            effort = torch.zeros((1, 1), device=self.device)
            if self.gravity_compensation and scale:
                if forces is None:
                    forces = self.robot.root_physx_view.get_gravity_compensation_forces()
                force_idx = torso_idx + (0 if self.robot.is_fixed_base else 6)
                effort[0, 0] = scale * forces[0, force_idx]
            self.robot.set_joint_effort_target(effort, joint_ids=[torso_idx])

    def _torso_joint_1_gravity_effort(self):
        """PhysX generalized effort opposing gravity; only the chosen DOF is applied."""
        idx = self.joint_name_to_index["astribot_torso_joint_1"]
        forces = self.robot.root_physx_view.get_gravity_compensation_forces()
        if not self.robot.is_fixed_base:
            idx += 6
        return forces[0, idx]

    def _sync_gripper_joints(self, joint_position_targets):
        """Convert compensated 0..100 commands to the six signed joint angles."""
        for side in ('left', 'right'):
            prefix = f'astribot_gripper_{side}_joint_'
            l1_idx = self.joint_name_to_index.get(prefix + 'L1')
            if l1_idx is None:
                continue

            command = joint_position_targets[l1_idx].clamp(0.0, 100.0)
            gripper_qpos = command * self._GRIPPER_RAD_PER_PERCENT
            for suffix, direction in self._GRIPPER_JOINT_DIRECTIONS.items():
                idx = self.joint_name_to_index.get(prefix + suffix)
                if idx is not None:
                    joint_position_targets[idx] = direction * gripper_qpos

    def _gripper_joint_to_percent(self, joint_name, value):
        direction = self._GRIPPER_JOINT_DIRECTIONS[joint_name.rsplit('_', 1)[-1]]
        return value / (direction * self._GRIPPER_RAD_PER_PERCENT)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_generation = getattr(self, 'reset_generation', 0) + 1

        self.scene.reset()
        self._restore_initial_state()
        self._publish_robot_placement()

        self.reset_flag = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _restore_initial_state(self):
        """Creation and reset consume the same resolved placement and joint pose."""
        if self.resolved_robot_placement is not None:
            import torch
            root = self.robot.data.default_root_state.clone()
            root[:, :7] = torch.tensor(
                self.resolved_robot_placement['articulation_root_pose_world'], device=self.device)
            root[:, 7:] = 0.
            self.robot.data.default_root_state.copy_(root)
            self.robot.write_root_pose_to_sim(root[:, :7])
            self.robot.write_root_velocity_to_sim(root[:, 7:])
            joints = self.robot.data.default_joint_pos.clone()
            velocity = self.robot.data.default_joint_vel.clone()
            velocity.zero_()
            self.robot.write_joint_state_to_sim(joints, velocity)
            self.robot.set_joint_position_target(joints)
            initial = self.resolved_robot_placement['initial_joint_positions']
            grippers = {item['part']: item for item in self.robot_profile_data['grippers'].values()}
            for part, interface in self.robot_dict.items():
                values = [initial[name] for name in self.robot_joint_map[part]]
                if part in grippers:
                    gripper = grippers[part]
                    measured = self.robot_joint_map[part][0]
                    values = [values[0] / (gripper['model_joints'][measured] * gripper['radians_per_percent'])]
                interface.controller_mode = 1
                interface.joint_position_command = values
                interface.joint_velocity_command = [0.] * len(values)
                interface.joint_torque_command = [0.] * len(values)
        from src.simulation_config.isaac_scene_objects import reset_scene_objects
        sampled = None
        if self.task_reset is not None:
            from src.simulation_config.task_reset import sample_task_objects
            sampled = sample_task_objects(self.task_scene, self.task_reset, self.np_random)
        self.last_reset_layout = sampled
        reset_scene_objects(self.scene, self.scene_object_keys, sampled=sampled)
        self.scene.write_data_to_sim()
        if self.resolved_robot_placement is not None:
            self.sim.physics_sim_view.update_articulations_kinematic()
            self.sim.forward()
        self.scene.update(dt=self.dt)

    def _publish_robot_placement(self):
        if self._placement_publisher is None:
            return
        from std_msgs.msg import String
        resolved = self.resolved_robot_placement
        indices, _ = self.robot.find_bodies(resolved['base_body'])
        if len(indices) != 1:
            raise ValueError('placement attestation requires a unique actual base body')
        actual = self.robot.data.body_state_w[0, indices[0], :7].detach().cpu().tolist()
        value = {'source': 'isaaclab.body_pose_w', 'observed_system_ns': time.time_ns(),
                 'base_body': resolved['base_body'], 'actual_base_pose_world': actual,
                 'declaration': resolved['declaration'],
                 'robot_control_sha256': resolved['robot_control_sha256'], 'resolved': resolved}
        self._placement_publisher.publish(String(data=json.dumps(value, allow_nan=False)))

    def evaluation_state(self):
        from src.simulation_config.isaac_scene_objects import read_scene_object_states
        from src.simulation_config.poses import relative_pose
        objects = read_scene_object_states(self.scene, self.scene_object_keys)
        if self.robot_profile_data is not None:
            indices, _ = self.robot.find_bodies(self.robot_profile_data['base_body'])
            if len(indices) != 1:
                raise ValueError('evaluation requires a unique actual robot base body')
            base = self.robot.data.body_state_w[0, indices[0], :7].detach().cpu().numpy()
            for value in objects.values():
                value['pose_base'] = relative_pose(base, value['pose_world']).tolist()
        return {'objects': objects, 'events': [], 'observations': {}}

    def _get_obs(self) -> np.ndarray:
        if self.object_names:
            return np.zeros(7)
        return np.zeros(7)

    def _get_info(self) -> dict:
        robot_info = {}

        for robot_name in self.robot_list:
            joint_names = self.robot_joint_map[robot_name]

            if 'gripper' in robot_name:
                joint_names = joint_names[:1]
                robot_info[robot_name] = self.get_joint_positions(joint_names)
            elif 'chassis' in robot_name:
                pose = self.get_chassis_pose()
                robot_info[robot_name] = pose
            else:
                robot_info[robot_name] = self.get_joint_positions(joint_names)

        return robot_info

    def render(self):
        if self.render_mode in ("human", "rgb_array"):
            for camera_name in self.camera_names:
                camera_data = self.render_single_camera(camera_name)
                if camera_data is not None:
                    self._publish_camera_data(camera_name, camera_data)
            if self.camera_names and not self._camera_output_logged:
                streams = ', '.join(
                    f"{name} ({width}x{height})"
                    for name, (width, height) in self.camera_resolutions.items()
                )
                astribot_simu_log(
                    f"RGB output started: {streams} "
                    f"(rgb8, mode={self.render_mode})"
                )
                self._camera_output_logged = True

    def setup_async_camera_rendering(self):
        # Isaac Sim render products must be read on the simulation thread.
        self.camera_rendering_active = False

    def close(self):
        astribot_simu_log("Closing Isaac Lab environment...")
        with self._video_lock:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
                self._video_stopped.set()
        self.shutdown_async_camera_rendering()
        # Disconnect Lab's timeline-stop callback before Kit closes the stage;
        # otherwise it enters the paused render loop during application shutdown.
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        IsaacLabAppManager.release_app()
        from src.simulation_config.isaac_scene_objects import close_scene_objects
        close_scene_objects(self.scene_object_keys)

    def get_joint_position(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_pos = self.robot.data.joint_pos[0, idx].cpu().item()

        if 'gripper' in joint_name:
            return self._gripper_joint_to_percent(joint_name, joint_pos)

        return joint_pos

    def get_joint_positions(self, names: List[str]) -> List[float]:
        return [self.get_joint_position(name) for name in names]

    def get_joint_velocity(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_vel = self.robot.data.joint_vel[0, idx].cpu().item()

        if 'gripper' in joint_name:
            return self._gripper_joint_to_percent(joint_name, joint_vel)

        return joint_vel

    def get_joint_velocities(self, names: List[str]) -> List[float]:
        return [self.get_joint_velocity(name) for name in names]

    def get_joint_acceleration(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_acc = self.robot.data.joint_acc[0, idx].cpu().item()

        if 'gripper' in joint_name:
            return self._gripper_joint_to_percent(joint_name, joint_acc)

        return joint_acc

    def get_joint_accelerations(self, names: List[str]) -> List[float]:
        return [self.get_joint_acceleration(name) for name in names]

    def get_joint_torque(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_index:
            return 0.0

        idx = self.joint_name_to_index[joint_name]
        joint_torque = self.robot.data.applied_torque[0, idx].cpu().item()

        return joint_torque

    def get_joint_torques(self, names: List[str]) -> List[float]:
        return [self.get_joint_torque(name) for name in names]

    def get_site_pose(self, site_name: str) -> np.ndarray:
        try:
            link_idx = self.robot.find_bodies(site_name)[0][0]
            pos = self.robot.data.body_pos_w[0, link_idx].cpu().numpy()
            quat = self.robot.data.body_quat_w[0, link_idx].cpu().numpy()
            return np.concatenate([pos, quat])
        except Exception:
            return np.array([0, 0, 0, 1, 0, 0, 0])

    def get_body_pose(self, body_name: str = None) -> np.ndarray:
        if body_name is None:
            return np.zeros(7)

        return self.get_site_pose(body_name)

    def get_chassis_pose(self) -> np.ndarray:
        pos = self.robot.data.root_pos_w[0].cpu().numpy()
        quat = self.robot.data.root_quat_w[0].cpu().numpy()
        return np.concatenate([pos, quat])

    def get_ft_sensor_data(self, robot_name: str) -> Optional[np.ndarray]:
        return None

    def get_reset_status(self) -> bool:
        if self.simulation_app is not None:
            return not self.simulation_app.is_running()
        return False

    def get_camera_image(self, camera_name: str = 'head_rgbd') -> Dict:
        return self.render_single_camera(camera_name)

    def render_single_camera(self, camera_name):
        if camera_name == 'head':
            camera_name = 'head_rgbd'
        if camera_name not in self.camera_names:
            return None

        rgb = self.scene[camera_name].data.output['rgb'][0]
        rgb = rgb.detach().cpu().numpy()
        # USD camera poses define image orientation; preserve sensor pixels.
        rgb = np.ascontiguousarray(rgb)
        return {
            'rgb_img': rgb,
            'depth_img': None,
            'point_cloud': None
        }
