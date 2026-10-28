#!/usr/bin/env bash
#SBATCH --partition=interact
#SBATCH --job-name=sjs_rad_extr
#SBATCH --array=1,2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=7:59:59
#SBATCH --output=/users/project1/pt01190/sjs/slurm/%A_%a.out
#SBATCH --error=/users/project1/pt01190/sjs/slurm/%A_%a.err




set -euo pipefail

SCRIPT_DIR="/users/project1/pt01190/sjs/texture_analysis"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

label="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

python extract_radiomics.py \
	--config "$SCRIPT_DIR/configs/config.yml" \
	--label "$label"
