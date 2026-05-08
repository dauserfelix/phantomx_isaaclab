# phantomx_thesis_env.py

from __future__ import annotations
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

from .phantomx_thesis_env_cfg import PhantomxThesisEnvCfg


class PhantomxThesisEnv(DirectRLEnv):
    cfg: PhantomxThesisEnvCfg

    def __init__(self, cfg: PhantomxThesisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_dof_idx, _ = self.robot.find_joints(".*")
        self.joint_default_pos = self.robot.data.default_joint_pos.clone()

        # cmd_vel commands: [lin_vel_x, lin_vel_y, yaw_rate]
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_joint_vel = torch.zeros(self.num_envs, len(self._joint_dof_idx), device=self.device)

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    # ---------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.cfg.terrain.num_envs = self.scene.num_envs
        self.cfg.terrain.env_spacing = self.cfg.scene.env_spacing
        self.terrain = TerrainImporter(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/skyLight", light_cfg)

    # ---------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.joint_default_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._joint_dof_idx)

    def _apply_action(self):
        self.robot.write_data_to_sim()

    # ---------------------------------------------------------------------

    def _get_observations(self) -> dict:
        joint_pos_offset = (
            self.robot.data.joint_pos[:, self._joint_dof_idx]
            - self.joint_default_pos[:, self._joint_dof_idx]
        )
        joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]

        obs = torch.cat([
            self.robot.data.root_lin_vel_b,       # 3
            self.robot.data.root_ang_vel_b,        # 3
            self.robot.data.projected_gravity_b,   # 3
            self.commands,                         # 3  ← cmd_vel
            joint_pos_offset,                      # 18
            joint_vel,                             # 18
            self.actions,                          # 18
        ], dim=-1)  # = 66

        return {"policy": obs}

    # ---------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_b = self.robot.data.root_lin_vel_b
        ang_vel_b = self.robot.data.root_ang_vel_b
        joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]

        # Tracking: wie gut folgt der Roboter dem cmd_vel?
        lin_vel_error = (lin_vel_b[:, 0] - self.commands[:, 0]) ** 2 \
                      + (lin_vel_b[:, 1] - self.commands[:, 1]) ** 2
        yaw_error     = (ang_vel_b[:, 2] - self.commands[:, 2]) ** 2

        r_lin_vel     = torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale
        r_yaw         = torch.exp(-yaw_error     / 0.25) * self.cfg.yaw_rate_reward_scale

        # Penalties (unverändert aus deiner Config)
        r_z_vel       = lin_vel_b[:, 2] ** 2 * self.cfg.z_vel_reward_scale
        r_ang_vel     = (ang_vel_b[:, 0] ** 2 + ang_vel_b[:, 1] ** 2) * self.cfg.ang_vel_reward_scale
        r_torque      = torch.sum(self.robot.data.applied_torque[:, self._joint_dof_idx] ** 2, dim=1) * self.cfg.joint_torque_reward_scale
        r_joint_accel = torch.sum((joint_vel - self._prev_joint_vel) ** 2, dim=1) * self.cfg.joint_accel_reward_scale
        r_action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1) * self.cfg.action_rate_reward_scale
        r_orientation = (self.robot.data.projected_gravity_b[:, 0] ** 2
                       + self.robot.data.projected_gravity_b[:, 1] ** 2) * self.cfg.flat_orientation_reward_scale
        r_height      = torch.exp(-(self.robot.data.root_pos_w[:, 2] - self.cfg.target_base_height) ** 2 / 0.01) * self.cfg.height_reward_scale
        r_alive       = torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward_scale

        self.prev_actions[:] = self.actions
        self._prev_joint_vel[:] = joint_vel

        return r_lin_vel + r_yaw + r_z_vel + r_ang_vel + r_torque + r_joint_accel + r_action_rate + r_orientation + r_height + r_alive

    # ---------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        too_low    = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        too_tilted = torch.norm(self.robot.data.projected_gravity_b[:, :2], dim=1) > self.cfg.termination_tilt
        return too_low | too_tilted, time_out

    # ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = torch.zeros_like(default_joint_pos)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0

        self._resample_commands(env_ids)

    # ---------------------------------------------------------------------

    def _resample_commands(self, env_ids: torch.Tensor):
        """Zufällige cmd_vel Befehle sampeln.
        Zur Inferenz einfach self.commands direkt überschreiben:
            env.commands[:, 0] = msg.linear.x
            env.commands[:, 1] = msg.linear.y
            env.commands[:, 2] = msg.angular.z
        """
        n = len(env_ids)
        self.commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.commands.lin_vel_x)
        self.commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.commands.lin_vel_y)
        self.commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.commands.yaw_rate)