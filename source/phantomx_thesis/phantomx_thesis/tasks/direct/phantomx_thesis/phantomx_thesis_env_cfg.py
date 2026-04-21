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

#  Kiesweg:
from isaaclab.terrains import TerrainImporterCfg


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

    randomize_terrain_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",  # Bei jedem Reset ändern
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 0.9),   # Große Variation!
            "dynamic_friction_range": (0.3, 0.7),  # Von rutschig bis griffig
            "restitution_range": (0.0, 0.2),       # Leichtes Federn
            "num_buckets": 128,
        },
    )

    # 🆕 Ground Contact Properties (am Boden selbst)
    randomize_ground_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ground"),  # Terrain selbst
            "static_friction_range": (0.5, 0.8),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
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
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=5,
            horizontal_scale=0.05,        # Kiesgröße
            vertical_scale=0.025,         # ±2.5cm Variation
            slope_threshold=None,
            use_cache=False,
            sub_terrains={
                "gravel": HfRandomUniformTerrainCfg(
                    size=(8.0, 8.0),
                    horizontal_scale=0.05,
                    vertical_scale=0.025,
                    border_width=0.5,
                    proportion=1.0,
                    noise_range=(0.01, 0.05),
                    noise_step=0.01,
                ),
            },
            curriculum=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.6,      # Rutschiger Kies
                dynamic_friction=0.4,     
                restitution=0.1,          # Leichtes Zurückfedern (Kies gibt nach)
            ),
            color_scheme="height",
            debug_vis=True,
        ),
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
