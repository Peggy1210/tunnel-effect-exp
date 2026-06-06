#!/bin/bash

models=("mlp12" "vgg19" "resnet34")
optimizers=("fullbatch_gd" "sgd" "sgd_momentum" "adam" "adamw" "muon")
datasets=("cifar10")

for model in "${models[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do

      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_${optimizer}_${dataset}
#SBATCH --account=ece556w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=q:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=output_${model}_${optimizer}_${dataset}.txt

/bin/hostname
python visualization.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset}
EOF

    done
  done
done