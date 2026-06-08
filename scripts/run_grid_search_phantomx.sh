#!/bin/bash

# -------------------------------
# CONFIGURATION
# -------------------------------
TASK="Isaac-Lab-Template-Phantomx-Thesis-Direct-v0"

# Parameter grids
LIN_VEL_SCALES=(5.0 10.0 15.0)
ALIVE_SCALES=(0.1 0.3 0.5)
TORQUE_SCALES=(-1e-5 -2e-5 -5e-5)
MOVEMENT_PENALTIES=(5.0 10.0 20.0)

# Optional: reduce episode length & envs for faster testing
OVERRIDES_FAST="env.episode_length_s=20.0 env.scene.num_envs=64"

# -------------------------------
# SETUP
# -------------------------------
RESULTS_DIR="/workspace/projects/phantomx_thesis/scripts/grid_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
EXPERIMENT_ID=1
TOTAL_COMBOS=$(( ${#LIN_VEL_SCALES[@]} * ${#ALIVE_SCALES[@]} * ${#TORQUE_SCALES[@]} * ${#MOVEMENT_PENALTIES[@]} ))

echo "=================================================="
echo "Starting Grid Search – Total experiments: $TOTAL_COMBOS"
echo "Results will be saved in: $RESULTS_DIR"
echo "=================================================="

for lin_vel in "${LIN_VEL_SCALES[@]}"; do
  for alive in "${ALIVE_SCALES[@]}"; do
    for torque in "${TORQUE_SCALES[@]}"; do
      for movement in "${MOVEMENT_PENALTIES[@]}"; do

        EXP_NAME="exp${EXPERIMENT_ID}_lv${lin_vel}_al${alive}_tq${torque}_mv${movement}"
        LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

        echo ""
        echo "[$(date +%H:%M:%S)] Running $EXP_NAME ($EXPERIMENT_ID/$TOTAL_COMBOS)"
        
        # Change to Isaac Lab directory and use isaaclab command
        cd /workspace/isaaclab
        
        # Run the training with overrides
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
          --task=$TASK \
          --headless \
          --experiment_name=$EXP_NAME \
          env.lin_vel_reward_scale=$lin_vel \
          env.alive_reward_scale=$alive \
          env.joint_torque_reward_scale=$torque \
          env.movement_penalty_scale=$movement \
          $OVERRIDES_FAST \
          2>&1 | tee "$LOG_FILE"

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
          echo "WARNING: Experiment $EXP_NAME failed. Check log."
        fi

        EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
      done
    done
  done
done

echo ""
echo "✅ Grid search completed! Results in $RESULTS_DIR"