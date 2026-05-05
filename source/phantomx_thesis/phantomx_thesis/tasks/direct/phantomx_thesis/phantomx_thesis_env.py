from __future__ import annotations

import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .phantomx_thesis_env_cfg import PhantomxThesisEnvCfg


class PhantomxThesisEnv(DirectRLEnv):
    cfg: PhantomxThesisEnvCfg

    def __init__(self, cfg: PhantomxThesisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs, 
            gym.spaces.flatdim(self.single_action_space), 
            device=self.device
        )
        self._previous_actions = torch.zeros(
            self.num_envs, 
            gym.spaces.flatdim(self.single_action_space), 
            device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands which the Agend should learn to track
        # Next steps: implement terminal input for this commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 🆕 Curriculum learning: track training progress
        self._training_iteration = 0

        # Get specific body indices for termination (all 6 tibias/feet)
        self._die_body_ids, _ = self._contact_sensor.find_bodies([
            "tibia_lf", "tibia_lm", "tibia_lr",  # Left feet
            "tibia_rf", "tibia_rm", "tibia_rr"   # Right feet
        ])

        self._has_stood_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "alive",  # 🆕 alive reward tracking
                "height_tracking",
                "movement_penalty",
            ]
        }

    # --------------------- SETUP ---------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------- ACTION ---------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions 
            + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # --------------------- OBSERVATIONS ---------------------
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ],
            dim=-1,
        )

        return {
            "policy": obs,
        }

    # --------------------- REWARDS ---------------------
    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking (exponential reward)
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), 
            dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        # yaw rate tracking (exponential reward)
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # z velocity penalty (should stay flat)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        
        # angular velocity x/y penalty (no roll/pitch)
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), 
            dim=1
        )
        
        # joint torques penalty (energy efficiency)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        
        # joint acceleration penalty (smooth movements)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        
        # action rate penalty (smooth control)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        
        # flat orientation penalty (stay upright)
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), 
            dim=1
        )

        # Base height tracking reward
        base_height = self._robot.data.root_pos_w[:, 2]

        height_error = torch.square(base_height - self.cfg.target_base_height)
        height_reward = torch.exp(-height_error / 0.02)
        
        # 🆕 Alive reward - critical for survival learning!
        alive_reward = torch.ones_like(lin_vel_error)

        #movement penalty for inactivity
        forward_speed = self._robot.data.root_lin_vel_b[:, 0]  # positiv = vorwärts, negativ = rückwärts
        is_moving_forward = forward_speed > self.cfg.movement_speed_x
        movement_penalty = (~is_moving_forward).float()  # Bestraft wenn nicht vorwärts bewegt

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "alive": alive_reward * self.cfg.alive_reward_scale * self.step_dt,  # 🆕

            "height_tracking": height_reward * self.cfg.height_reward_scale,
            "movement_penalty": -movement_penalty * self.cfg.movement_penalty_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    # --------------------- TERMINATION ---------------------
    def _get_dones(self):
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Base height
        base_height = self._robot.data.root_pos_w[:, 2]

        # Tilt (wie stark gekippt)
        gravity = self._robot.data.projected_gravity_b
        tilt = torch.sum(torch.square(gravity[:, :2]), dim=1)

        # Threshold (anpassen!)
        max_tilt = 0.1  # ~ ca. 45–60° Neigung

        # Termination:
        died = (
            (base_height < 0.05) |
            (base_height > 0.20) |
            (tilt > max_tilt)
        )

        return died, time_out
        
    
        
    # --------------------- RESET ---------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
            
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid spikes in training
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, 
                high=int(self.max_episode_length)
            )
            
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        #aufstehen antraineren- variable rücksetzen
        self._has_stood_up[env_ids] = False
        
        # 🆕 CURRICULUM LEARNING für Commands
        # Starts easy (slow speeds), gets harder as training progresses
        self._training_iteration += 1
        
        # Curriculum stages:
        # 0-200 iterations: only standing (commands = 0)
        # 200-500: slow forward walk
        # 500+: full random commands
        if self._training_iteration < 150:
            # Stage 1: Learn to stand
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 0.3  # forward
        elif self._training_iteration < 300:
            # Stage 2: Learn slow forward walk
            self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 0.6  # forward
            self._commands[env_ids, 1] = 0.0  # no sideways
            self._commands[env_ids, 2] = 0.0  # no turning
        else:
            # Stage 3: Full curriculum - sample from increasing ranges
            curriculum_factor = min(1.0, (self._training_iteration - 500) / 1000)
            max_vel = 0.3 + curriculum_factor * 0.7  # grows from 0.3 to 1.0
            
            self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
                -max_vel, max_vel
            )
        
        # Reset robot state with small randomization for robustness
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        # 🆕 Small initial joint randomization
        joint_pos += torch.randn_like(joint_pos) * 0.10   #default: 0.05
        
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # 🆕 Small initial height variation
        default_root_state[:, 2] += torch.randn(len(env_ids), device=self.device) * 0.01
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging episode statistics
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
            
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        

    def get_IO_descriptors(self) -> dict:
        return {
            "observations": {
                "policy": [
                    {"name": "root_lin_vel_b",          "size": 3,  "description": "Root linear velocity in body frame (x, y, z)"},
                    {"name": "root_ang_vel_b",           "size": 3,  "description": "Root angular velocity in body frame (roll, pitch, yaw)"},
                    {"name": "projected_gravity_b",      "size": 3,  "description": "Projected gravity vector in body frame"},
                    {"name": "commands",                 "size": 3,  "description": "Velocity commands (vx, vy, yaw_rate)"},
                    {"name": "joint_pos_rel",            "size": 18, "description": "Joint positions relative to default pose"},
                    {"name": "joint_vel",                "size": 18, "description": "Joint velocities"},
                    {"name": "actions",                  "size": 18, "description": "Previous actions"},
                ]
            },
            "actions": [
                {
                    "name": "joint_position_targets",
                    "size": 18,
                    "description": "Joint position targets (scaled deviation from default + default_joint_pos)",
                    "scale": self.cfg.action_scale,
                }
            ],
            "articulations": {
                "robot": {
                    "num_joints": self._robot.num_joints,
                    "joint_names": self._robot.joint_names,
                }
            },
            "scene": {
                "num_envs": self.num_envs,
                "terrain": str(type(self._terrain).__name__),
                "target_base_height": self.cfg.target_base_height,
            },
        }