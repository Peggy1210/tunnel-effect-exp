#!/bin/bash
#SBATCH --job-name=compress_tar_gz
#SBATCH --account=ece556w26_class
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=compress_%j.out

set -euo pipefail

/bin/hostname
echo "Running on ${SLURM_CPUS_PER_TASK} CPUs"

# 如果系統有 module
module load pigz 2>/dev/null || true

for d in */; do
    if [ -d "$d/checkpoints" ]; then
        echo "Compressing checkpoints in $d..."
        tar -cf - -C "$d" checkpoints | pigz -p "$SLURM_CPUS_PER_TASK" > "${d%/}/checkpoints.tar.gz"
    fi
done