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
    episode_length_s = 20.0
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
    # ROBOT
    # =====================================================
    robot: ArticulationCfg = PHANTOMX_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    target_base_height = 0.10    # 120 mm
    min_movement_threshold = 0.05   # 5 cm/s
    

    # =====================================================
    # REWARD SCALES - TUNED FOR HEXAPOD LOCOMOTION
    # =====================================================
    # 🎯 TRACKING REWARDS (positive)
    lin_vel_reward_scale = 3.0      
    yaw_rate_reward_scale = 2.0     # 🔧 Reduced from 1.0 - yaw weniger wichtig

    height_reward_scale = 0.2   # Stärke des Rewards (tunable)
    
    
    # 🚫 PENALTIES (negative)
    z_vel_reward_scale = -2.0       # Bleib flach
    ang_vel_reward_scale = -2.5    # Kein Roll/Pitch
    joint_torque_reward_scale = -2e-5   # 🔧 Leicht erhöht - energie-effizienz wichtiger
    joint_accel_reward_scale = -2.5e-7  # 🔧 Erhöht - sanfte Bewegungen fördern
    action_rate_reward_scale = -0.02    # Keine ruckartigen Actions
    flat_orientation_reward_scale = -3.0  # 🔧 Reduced from -5.0 - zu harsh

    movement_penalty_scale = 3.0
    
    # 🆕 SURVIVAL REWARD (critical!)
    alive_reward_scale = 0.3  # Konstante Belohnung fürs Überleben
    
    # =====================================================
    # TERMINATION THRESHOLDS - RELAXED FOR LEARNING
    # =====================================================
    # 🔧 WICHTIG: Diese Werte bestimmen wann eine Episode abbricht
    # Zu streng = Robot kann nicht lernen
    # Zu locker = Robot lernt schlechte Strategien
    
    # Height threshold - unter dieser Höhe stirbt der Robot
    # Default: 0.02m (2cm) - SEHR niedrig, nur für komplett umgefallen
    # 🔧 Erhöhe das schrittweise wenn mean_episode_length > 200 erreicht ist:
    #    Phase 1 (jetzt): 0.05m - sehr tolerant, lernt Stabilität
    #    Phase 2 (bei >100 ep_len): 0.07m - etwas strenger
    #    Phase 3 (bei >200 ep_len): 0.09m - finale Schwelle
    termination_height = 0.03
    
    # Tilt threshold - maximale Neigung bevor Episode abbricht
    # projected_gravity_b[:, :2] squared sum
    # Bei 0° upright: ~0.0
    # Bei 45° tilt: ~0.5
    # Bei 90° (auf Seite): ~1.0
    # 🔧 Aktuelle Einstellung: 1.0 = sehr tolerant (fast auf der Seite)
    #    Reduziere das später auf 0.8 → 0.6 → 0.5 für bessere Performance
    termination_tilt = -0.5
    
    # Angle reference (kept for compatibility, not actively used)
    max_tilt_angle_deg = 45.0


# =====================================================
# TRAINING PROGRESSION GUIDE
# =====================================================
# 
# 📊 Monitoring:
# - Mean episode length sollte wachsen: 13 → 50 → 100 → 200+
# - time_out sollte steigen: 0.0 → 0.1 → 0.5 → 0.9+
# - Mean reward sollte positiv werden und wachsen
#
# 🎓 Curriculum Stages (automatisch in env.py):
# - Stage 1 (iter 0-200): Nur stehen lernen (commands=0)
# - Stage 2 (iter 200-500): Langsames Vorwärtsgehen
# - Stage 3 (iter 500+): Volle Befehle mit ansteigender Schwierigkeit
#
# 🔧 Manual Tuning Schedule:
# 
# PHASE 1 - SURVIVAL (jetzt):
# ✓ termination_height = 0.05 (sehr tolerant)
# ✓ termination_tilt = 1.0 (sehr tolerant)
# ✓ alive_reward_scale = 0.5 (hoch)
# Ziel: mean_episode_length > 100
#
# PHASE 2 - BASIC LOCOMOTION (wenn ep_len > 100):
# - termination_height = 0.07
# - termination_tilt = 0.8
# - alive_reward_scale = 0.3
# - lin_vel_reward_scale = 8.0 (erhöhen)
# Ziel: mean_episode_length > 200, time_out > 0.3
#
# PHASE 3 - REFINEMENT (wenn ep_len > 200):
# - termination_height = 0.09
# - termination_tilt = 0.6
# - alive_reward_scale = 0.1
# - lin_vel_reward_scale = 10.0
# - joint_accel_reward_scale = -5e-7 (glattere Bewegungen)
# Ziel: time_out > 0.7, smooth gait
#
# PHASE 4 - POLISHING (wenn time_out > 0.7):
# - termination_tilt = 0.5
# - action_rate_reward_scale = -0.02 (sanftere Actions)
# - Füge weitere Rewards hinzu (z.B. foot clearance, symmetrie)
# =====================================================