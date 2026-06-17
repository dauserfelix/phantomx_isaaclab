from __future__ import annotations

import math
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
        

        # Get specific body indices for termination (all 6 tibias/feet)
        self._die_body_ids, _ = self._contact_sensor.find_bodies([
            "tibia_lf", "tibia_lm", "tibia_lr",  # Left feet
            "tibia_rf", "tibia_rm", "tibia_rr"   # Right feet
        ])

        self._has_stood_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # CPG — Tripod-Gang
        self._cpg_phase = torch.zeros(self.num_envs, device=self.device)

        # Tripod A (LF, RM, LR): Phase-Offset = 0
        # Tripod B (RF, LM, RR): Phase-Offset = π
        _A_coxa,  _ = self._robot.find_joints(["j_c1_lf",    "j_c1_rm",    "j_c1_lr"])
        _B_coxa,  _ = self._robot.find_joints(["j_c1_rf",    "j_c1_lm",    "j_c1_rr"])
        _A_femur, _ = self._robot.find_joints(["j_thigh_lf", "j_thigh_rm", "j_thigh_lr"])
        _B_femur, _ = self._robot.find_joints(["j_thigh_rf", "j_thigh_lm", "j_thigh_rr"])
        _A_tibia, _ = self._robot.find_joints(["j_tibia_lf", "j_tibia_rm", "j_tibia_lr"])
        _B_tibia, _ = self._robot.find_joints(["j_tibia_rf", "j_tibia_lm", "j_tibia_rr"])
        n = self._robot.num_joints  # 18

        # Phase-Offset (18,): 0 = Gruppe A, π = Gruppe B
        self._cpg_phase_offset = torch.zeros(n, device=self.device)
        self._cpg_phase_offset[list(_B_coxa) + list(_B_femur) + list(_B_tibia)] = math.pi

        # Gelenk-Typ-Masken (18,)
        self._cpg_coxa_mask  = torch.zeros(n, device=self.device)
        self._cpg_femur_mask = torch.zeros(n, device=self.device)
        self._cpg_tibia_mask = torch.zeros(n, device=self.device)
        self._cpg_coxa_mask [list(_A_coxa)  + list(_B_coxa)]  = 1.0
        self._cpg_femur_mask[list(_A_femur) + list(_B_femur)] = 1.0
        self._cpg_tibia_mask[list(_A_tibia) + list(_B_tibia)] = 1.0

        # Coxa-Vorzeichen: alle Coxa-Gelenke drehen um Körper-Z-Achse, aber
        # rechte Beine: positiv = vorwärts; linke Beine: positiv = rückwärts.
        # → linke Beine brauchen Vorzeichenumkehr im CPG.
        _left_coxa,  _ = self._robot.find_joints(["j_c1_lf", "j_c1_lm", "j_c1_lr"])
        _right_coxa, _ = self._robot.find_joints(["j_c1_rf", "j_c1_rm", "j_c1_rr"])
        self._cpg_coxa_sign = torch.zeros(n, device=self.device)
        self._cpg_coxa_sign[list(_right_coxa)] = +1.0
        self._cpg_coxa_sign[list(_left_coxa)]  = -1.0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "height_tracking",
                "flat_orientation_l2",
                "dof_torques_l2",
                "action_rate_l2",
                "alive",
                "foot_contact",
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
        # Motor-Strength-Curriculum (nur effort_limit, stiffness fest):
        # Ramp über 700k Steps (58% des Trainings) → danach 500k Steps mit realen Motoren.
        progress = min(1.0, self.common_step_counter / 700_000)
        effort_limit = 3.82 + (1.912 - 3.82) * progress   # 3.82 → 1.912 Nm

        actuator = self._robot.actuators["all_joints"]
        actuator.stiffness[:] = 8.0    # fest: realistisch für AX-12-Klasse

        # Grace-Period (0.5s nach Spawn): volles Drehmoment damit Roboter in Default-Pose steht.
        # Unabhängig von episode_length_buf (wird bei Bulk-Reset zufällig gesetzt).
        grace_steps = int(0.5 / self.step_dt)
        in_grace = self._steps_since_reset < grace_steps
        self._steps_since_reset += 1
        effort_tensor = actuator.effort_limit.clone().fill_(effort_limit)
        effort_tensor[in_grace] = 3.82
        actuator.effort_limit[:] = effort_tensor

        self._actions = actions.clone()
        self._actions[in_grace] = 0.0  # Observation konsistent: Policy hat "nichts" gemacht

        # CPG Phase vorrücken
        self._cpg_phase.add_(2.0 * math.pi * self.cfg.cpg_freq * self.step_dt)
        self._cpg_phase.fmod_(2.0 * math.pi)

        # q_cpg berechnen
        phase   = self._cpg_phase.unsqueeze(1) + self._cpg_phase_offset  # (N, 18)
        sin_phi = torch.sin(phase)
        swing   = torch.clamp(sin_phi, min=0.0)
        q_cpg   = (
              self._cpg_coxa_mask  * self.cfg.cpg_A_coxa * self._cpg_coxa_sign * sin_phi
            + self._cpg_femur_mask * self.cfg.cpg_A_femur                       * swing
            + self._cpg_tibia_mask * (-self.cfg.cpg_A_tibia)                    * swing
        )

        # q_target = q_default + q_gait + scale * Δq_policy
        q_def = self._robot.data.default_joint_pos
        self._processed_actions = q_def + q_cpg + self.cfg.action_scale * self._actions
        self._processed_actions = torch.clamp(
            self._processed_actions,
            q_def - self.cfg.joint_pos_limit,
            q_def + self.cfg.joint_pos_limit,
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # --------------------- OBSERVATIONS ---------------------
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # Sensor-Messungen mit Rauschen (Sim-to-Real: modelliert Encoder/IMU-Noise)
        lin_vel  = self._robot.data.root_lin_vel_b  + torch.randn_like(self._robot.data.root_lin_vel_b)  * 0.01
        ang_vel  = self._robot.data.root_ang_vel_b  + torch.randn_like(self._robot.data.root_ang_vel_b)  * 0.01
        gravity  = self._robot.data.projected_gravity_b + torch.randn_like(self._robot.data.projected_gravity_b) * 0.01
        jpos_rel = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) + torch.randn(self.num_envs, self._robot.num_joints, device=self.device) * 0.01
        jvel     = self._robot.data.joint_vel   + torch.randn(self.num_envs, self._robot.num_joints, device=self.device) * 0.05

        obs = torch.cat(
            [
                lin_vel,                                                     # 3
                ang_vel,                                                     # 3
                gravity,                                                     # 3
                self._commands,                                              # 3  (kein Noise)
                jpos_rel,                                                    # 18
                jvel,                                                        # 18
                self._actions,                                               # 18 (kein Noise)
                torch.sin(self._cpg_phase).unsqueeze(1),                    # 1  (kein Noise)
                torch.cos(self._cpg_phase).unsqueeze(1),                    # 1
            ],
            dim=-1,
        )  # total: 68

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

        # joint torques penalty (energy efficiency)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

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
        height_reward = torch.exp(-height_error / 0.05)   # lockerer als vorher (0.02 → 0.05)

        # Alive reward
        alive_reward = torch.ones_like(lin_vel_error)

        # Foot contact reward — bonus for stable tripod support base (≥3 feet on ground)
        foot_forces = self._contact_sensor.data.net_forces_w[:, self._die_body_ids, :]  # (num_envs, 6, 3)
        num_feet_in_contact = (torch.norm(foot_forces, dim=-1) > 1.0).float().sum(dim=-1)  # (num_envs,)
        foot_contact_reward = torch.clamp(num_feet_in_contact / 3.0, max=1.0)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "height_tracking":      height_reward         * self.cfg.height_reward_scale * self.step_dt,
            "flat_orientation_l2":  flat_orientation      * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "dof_torques_l2":       joint_torques         * self.cfg.joint_torque_reward_scale * self.step_dt,
            "action_rate_l2":       action_rate           * self.cfg.action_rate_reward_scale * self.step_dt,
            "alive":                alive_reward          * self.cfg.alive_reward_scale * self.step_dt,
            "foot_contact":         foot_contact_reward   * self.cfg.foot_contact_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward
    # --------------------- TERMINATION ---------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_height = self._robot.data.root_pos_w[:, 2]
        gravity = self._robot.data.projected_gravity_b
        tilt = torch.sum(torch.square(gravity[:, :2]), dim=1)

        # 1s Grace-Period: Roboter hat Zeit zum Stabilisieren nach dem Reset
        # grace_period = self.episode_length_buf > (0.5 / self.step_dt)  # 0.5s = 25 steps

        foot_forces = self._contact_sensor.data.net_forces_w[:, self._die_body_ids, :]  # (envs, 6, 3)
        num_feet_in_contact = (torch.norm(foot_forces, dim=-1) > 1.0).float().sum(dim=-1)  # (envs,)

        died = (
            (base_height < self.cfg.termination_height) |
            (base_height > 0.35) |
            (tilt > self.cfg.termination_tilt)
        )

        return died, time_out
        
    
        
    # --------------------- RESET ---------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
            
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.common_step_counter < 2:
            print(f"env_origins[:5]:\n{self._terrain.env_origins[:5]}")
            print(f"unique origins: {self._terrain.env_origins.unique(dim=0).shape[0]} / {self.num_envs}")

        if len(env_ids) == self.num_envs and self.common_step_counter > 0:
            # Spread out resets during training to avoid synchronized resets.
            # Wird bei common_step_counter == 0 (Play-Start oder Training-Initialisierung)
            # übersprungen, damit der Roboter nicht sofort mit aktivem grace_period startet.
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length)
            )
            
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._cpg_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * math.pi

        self._has_stood_up[env_ids] = False

        # CURRICULUM LEARNING: basiert auf common_step_counter (echte Simulations-Steps),
        # nicht auf Reset-Zähler. Mit decimation=4, dt=1/200 → step_dt=0.02s.
        # 50 000 Steps ≈ 1000s Sim-Zeit; 150 000 Steps ≈ 3000s.
        # Velocity-Curriculum — kalibriert auf 1 200 000 Gesamt-Steps:
        #   Stage 1 (    0 – 100k): stehen lernen, kein Command           (  8%)
        #   Stage 2 (100k – 350k):  langsam vorwärts, 0–0.5 m/s          ( 21%)
        #   Stage 3 (350k –   1M):  voller Bereich, wächst auf ±1.0 m/s  ( 54%)
        #                           Ramp endet bei 350k + 700k = 1050k   → voll ab ~1.05M
        #   Danach (1.05M – 1.2M):  volles ±1.0 m/s, reale Motoren       ( 12%)
        steps = self.common_step_counter
        if steps < 100_000:
            self._commands[env_ids] = 0.0
        elif steps < 350_000:
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 0.5
        else:
            curriculum_factor = min(1.0, (steps - 350_000) / 700_000)
            max_vel = 0.3 + curriculum_factor * 0.7   # 0.3 → 1.0 m/s
            self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
                -max_vel, max_vel
            )
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos += torch.randn_like(joint_pos) * (math.pi / 18)  # ±10° wie PyBullet-Referenz
        
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
                    {"name": "cpg_phase_sin_cos",         "size": 2,  "description": "sin/cos of CPG phase (encodes tripod gait cycle)"},
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