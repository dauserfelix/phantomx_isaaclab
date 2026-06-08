# phantomx_thesis_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab_assets.robots.phantomx import PHANTOMX_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),  # 🔧 Mehr Variation für Robustheit
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="MP_BODY"),
            "mass_distribution_params": (-0.5, 0.5),  # 🔧 Kleinere Variation am Anfang
            "operation": "add",
        },
    )
    


@configclass
class PhantomxThesisEnvCfg(DirectRLEnvCfg):
    # =====================================================
    # ENVIRONMENT SETUP
    # =====================================================
    episode_length_s = 40.0
    decimation = 4
    action_scale = 0.5  # 🔧 Reduziert von 1.0 - kleinere Actions für Stabilität
    action_space = 18  # PhantomX: 6 legs × 3 joints = 18 DOF
    
    # Observation space: 
    #   root_lin_vel_b (3) + root_ang_vel_b (3) + projected_gravity_b (3)
    #   + commands (3) + joint_pos_offset (18) + joint_vel (18) + actions (18)
    #   Total = 66
    observation_space = 66
    state_space = 0
    
    obs_groups = {
        "actor": "policy",
        "critic": "policy",
    }

    # =====================================================
    # SIMULATION
    # =====================================================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # =====================================================
    # SCENE
    # =====================================================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200, 
        env_spacing=2.5, 
        replicate_physics=True
    )

    # =====================================================
    # EVENTS (RANDOMIZATION)
    # =====================================================
    events: EventCfg = EventCfg()

    # =====================================================
    # ROBOT Movement Params
    # =====================================================
    robot: ArticulationCfg = PHANTOMX_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    target_base_height = 0.10    # 120 mm
    movement_speed_x = 0.10   # 10 cm/s
    yaw_rotation_speed_x = 0.0   # 0 rad/s 
    

    # =====================================================
    # REWARD SCALES - TUNED FOR HEXAPOD LOCOMOTION
    # =====================================================
    # 🎯 TRACKING REWARDS (positive)
    lin_vel_reward_scale = 10.0      
    yaw_rate_reward_scale = 4.0     # 🔧 Reduced from 1.0 - yaw weniger wichtig

    height_reward_scale = 0.1   # Stärke des Rewards (tunable)
    
    
    # 🚫 PENALTIES (negative)
    z_vel_reward_scale = -2.0       # Bleib flach
    ang_vel_reward_scale = -5      # Kein Roll/Pitch
    joint_torque_reward_scale = -2e-5   # 🔧 Leicht erhöht - energie-effizienz wichtiger
    joint_accel_reward_scale = -2.5e-7  # 🔧 Erhöht - sanfte Bewegungen fördern
    action_rate_reward_scale = -0.02    # Keine ruckartigen Actions
    flat_orientation_reward_scale = -3.0  # 🔧 Reduced from -5.0 - zu harsh

    movement_penalty_scale = 10.0
    
    # 🆕 SURVIVAL REWARD (critical!)
    alive_reward_scale = 0.3  # Konstante Belohnung fürs Überleben
    
    # =====================================================
    # TERMINATION THRESHOLDS - RELAXED FOR LEARNING
    # =====================================================

    termination_height = 0.03

    termination_tilt = 0.5