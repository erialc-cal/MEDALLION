#!/bin/bash
#SBATCH --job-name=mac-sweep
#SBATCH --account=stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --chdir=/burg-archive/stats/users/cmh2277/MEDALLION/
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cmh2277@columbia.edu

set -euo pipefail

source "/burg-archive/stats/users/cmh2277/miniforge3/etc/profile.d/conda.sh"
conda activate medal_env

cd /burg-archive/stats/users/cmh2277/MEDALLION/encoder_FI

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

export MEDAL_CPUS_PER_TRIAL=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MEDAL_SMOKE_TEST=0
export RAY_TQDM_PATCH_PRINT=0 # Ray replaces Python's built in print with logging wrapper that raises a Numba error if not set to 0

# Do not overwrite CUDA_VISIBLE_DEVICES; Slurm and Ray manage it.

echo "Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

which python
python -c "import medal; print('MEDAL:', medal.__file__)"
nvidia-smi

srun --ntasks=1 python -u train_macaque.py