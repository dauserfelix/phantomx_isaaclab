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
            "static_friction_range": (0.5, 1.0),  # 🔧 Rutschiger für Kies
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="MP_BODY"),
            "mass_distribution_params": (-0.5, 0.5),
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


@configclass
class PhantomxThesisEnvCfg(DirectRLEnvCfg):
    # =====================================================
    # ENVIRONMENT SETUP
    # =====================================================
    episode_length_s = 40.0
    decimation = 4
    action_scale = 0.5
    action_space = 18
    
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
    
    # =====================================================
    # 🆕 TERRAIN - EINFACHE VERSION DIE FUNKTIONIERT
    # =====================================================
    # Für jetzt: Flacher Boden mit variabler Friction
    # Die Terrain-Randomisierung passiert über EventCfg
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # Einfach, funktioniert garantiert
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,      # Mittlere Friction (wie Kies)
            dynamic_friction=0.5,     # Rutschiger beim Bewegen
            restitution=0.05,         # Leichtes Federn
        ),
        debug_vis=False,
    )
    
    # ALTERNATIVE: Nutze vordefinierte Rough Terrains
    # Kommentiere "plane" aus und aktiviere dies:
    """
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.copy(),  # Nutze vordefinierte Config
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,
            dynamic_friction=0.5,
            restitution=0.05,
        ),
        debug_vis=False,
    )
    """
    
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
    target_base_height = 0.10
    movement_speed_x = 0.10
    yaw_rotation_speed_x = 0.0
    
    # =====================================================
    # REWARD SCALES
    # =====================================================
    lin_vel_reward_scale = 10.0
    yaw_rate_reward_scale = 4.0
    height_reward_scale = 0.1
    
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -5
    joint_torque_reward_scale = -2e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.02
    flat_orientation_reward_scale = -3.0
    movement_penalty_scale = 10.0
    
    alive_reward_scale = 0.3
    
    # =====================================================
    # TERMINATION THRESHOLDS
    # =====================================================
    termination_height = 0.03
    termination_tilt = 0.5