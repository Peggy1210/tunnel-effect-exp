#!/bin/bash
# #SBATCH --job-name=compress_tar_gz
# #SBATCH --account=ece556w26_class
# #SBATCH --partition=standard
# #SBATCH --time=8:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=20G
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output=compress_%j.out

# set -euo pipefail

# /bin/hostname
# echo "Running on ${SLURM_CPUS_PER_TASK} CPUs"

# # 如果系統有 module
# module load pigz 2>/dev/null || true

# for d in */; do
#     if [ -d "$d/analysis" ]; then
#         echo "Compressing analysis in $d..."
#         tar -cf - -C "$d" analysis | pigz -p "$SLURM_CPUS_PER_TASK" > "${d%/}/analysis.tar.gz"
#     fi
# done

#SBATCH --job-name=compress_models
#SBATCH --account=ece556w26_class
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=compress_%A_%a.out

set -euo pipefail

models=(resnet mlp12 vgg19)
prefix="${models[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Prefix: $prefix"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module load pigz 2>/dev/null || true

for d in ${prefix}*/; do
    [ -d "$d" ] || continue

    if [ -d "$d/analysis" ]; then
        echo "Compressing $d/analysis..."
        tar -cf - -C "$d" analysis | pigz -p "$SLURM_CPUS_PER_TASK" > "${d%/}/analysis.tar.gz"
    fi
done