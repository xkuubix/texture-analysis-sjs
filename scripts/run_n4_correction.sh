#!/usr/bin/env bash
#SBATCH --partition=interact
#SBATCH --job-name=sjs_n4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=7:59:59
#SBATCH --output=/users/project1/pt01190/sjs/slurm/%A.out
#SBATCH --error=/users/project1/pt01190/sjs/slurm/%A.err


SCRIPT_DIR="/users/project1/pt01190/sjs/texture_analysis"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/.venv/bin/activate"

python n4_correct.py \
	--config "$SCRIPT_DIR/configs/config.yml" \
