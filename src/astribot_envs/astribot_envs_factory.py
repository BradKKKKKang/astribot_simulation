#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Tony Wang, tonywang@astribot.com
# -----------------------------------------------------------------------------

"""
File: astribot_envs_factory.py
Brief: Factory for simulation env
"""

import os
import time
import yaml
import importlib

import threading
import gymnasium
from src.astribot_envs.simulation_timing import advance_periodic_deadline

ros_version = os.getenv('ROS_VERSION')

if ros_version=='1':
    import rospy
elif ros_version=='2':
    import rclpy
    
class AstribotEnvsFactory:
    def __init__(self):
        self.astribot_simu_env=None
        self.stop_event = threading.Event()

    def create_simulation_env(self,data):
        module_name = f"src.astribot_envs.astribot_{data['simulator_type'].lower()}_env"
        class_name = f"Astribot{data['simulator_type']}Env"

        try:
            module = importlib.import_module(module_name)
            _ = getattr(module, class_name)  # Trigger module registration
            gym_env_name=f"astribot_envs/{class_name}-v0"
            self.astribot_simu_env=gymnasium.make(gym_env_name,param=data)

            # Isaac/rclpy may install their own handlers while constructing the app.
            if hasattr(self, 'install_stop_handlers'):
                self.install_stop_handlers()

            # Check if this is Isaac Lab - it must run in main thread
            is_isaaclab = hasattr(self.astribot_simu_env.unwrapped, 'simulation_app')

            if is_isaaclab:
                from src.simu_utils.simu_common_tools import astribot_simu_log
                astribot_simu_log("Isaac Lab detected - will run in main thread (GUI requires main thread)")
                # Don't start background thread for Isaac Lab
                # Main thread will be blocked in simu_env_loop()
                self.simu_thread = None
                # Run directly in main thread (this will block)
                self.simu_env_loop()
                return None  # Won't return until simulation ends
            else:
                # Other simulators can run in background thread
                self.simu_thread = threading.Thread(target=self.simu_env_loop, args=())
                self.simu_thread.start()
                return self.simu_thread
        except ModuleNotFoundError:
            raise ValueError(f"Unknown simulator type: {data['simulator_type']}. Module: {module_name} not found.")
        except AttributeError:
            raise ValueError(f"Class {class_name} not found in {module_name}.")
    
    @staticmethod
    def load_yaml_file(yaml_file_path):
        with open(yaml_file_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
            astribot_simu_root = os.getenv('ASTRIBOT_SIMU_ROOT')
            data['model_path'] = astribot_simu_root + data['model_path']
        return data

    def running(self):
        if ros_version=='1':
            if not hasattr(self, 'rate'):
                self.rate = rospy.Rate(50)
            return not rospy.is_shutdown()
        elif ros_version=='2':
            if not hasattr(self, 'rate'):
                self.rate = self.astribot_simu_env.unwrapped.node.create_rate(50)
            if not hasattr(self, 'spin_thread'):
                self.executor = rclpy.executors.SingleThreadedExecutor()
                self.executor.add_node(self.astribot_simu_env.unwrapped.node)
                self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
                self.spin_thread.start()
            return rclpy.ok()
        
    def simu_env_loop(self):
        observation, info = self.astribot_simu_env.reset()
        self._evaluate_state(reset=True)

        # Check if this is Isaac Lab environment
        is_isaaclab = hasattr(self.astribot_simu_env.unwrapped, 'simulation_app')

        if is_isaaclab:
            # Isaac Lab specific loop - use app.is_running() instead of ROS rate
            # The simulation_app must control the loop timing
            from src.simu_utils.simu_common_tools import astribot_simu_log
            astribot_simu_log("Using Isaac Lab loop mode (app controls timing)")

            # For ROS2, we still need to setup the executor (but not rate)
            if ros_version == '2':
                if not hasattr(self, 'spin_thread'):
                    self.executor = rclpy.executors.SingleThreadedExecutor()
                    self.executor.add_node(self.astribot_simu_env.unwrapped.node)
                    self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
                    self.spin_thread.start()
                    astribot_simu_log("ROS2 executor started in background thread")

            app = self.astribot_simu_env.unwrapped.simulation_app
            control_hz = float(os.getenv("ASTRIBOT_ISAAC_CONTROL_HZ", "0"))
            if control_hz < 0:
                raise ValueError("ASTRIBOT_ISAAC_CONTROL_HZ must be non-negative")
            period_s = 1.0 / control_hz if control_hz else 0.0
            next_step = time.monotonic_ns()
            period_ns = int(period_s * 1e9) if period_s else 0
            while app.is_running() and not self.stop_event.is_set():
                if period_s:
                    remaining_ns = next_step - time.monotonic_ns()
                    if remaining_ns > 0:
                        self.stop_event.wait(remaining_ns / 1e9)
                    if self.stop_event.is_set():
                        break
                action = self.astribot_simu_env.action_space.sample()
                observation, reward, terminated, truncated, info = self.astribot_simu_env.step(action)
                self._evaluate_state()
                if (terminated or truncated) and getattr(self, 'evaluator', None) is not None:
                    self.stop_event.set()
                if period_s:
                    # Keep absolute deadlines; skip missed periods after overruns.
                    next_step = advance_periodic_deadline(
                        next_step, time.monotonic_ns(), period_ns,
                    )
                # Physics dt stays fixed; this optional pacing affects wall time only.
        else:
            # Normal loop for MuJoCo, Genesis, ManiSkill
            while not self.stop_event.is_set() and self.running():
                action = self.astribot_simu_env.action_space.sample()
                observation, reward, terminated, truncated, info = self.astribot_simu_env.step(action)
                self._evaluate_state()
                if (terminated or truncated) and getattr(self, 'evaluator', None) is not None:
                    self.stop_event.set()
                self.rate.sleep()

        # The entrypoint owns cleanup after stopping the ROS executor.

    def _evaluate_state(self, *, reset=False):
        evaluator = getattr(self, 'evaluator', None)
        if evaluator is not None and evaluator.observe(self.astribot_simu_env.unwrapped, reset=reset):
            self.stop_event.set()

