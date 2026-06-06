models=("mlp12" "vgg19" "resnet34")
optimizers=("fullbatch_gd" "sgd" "sgd_momentum" "adam" "adamw" "muon")
datasets=("cifar10")

for model in "${models[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
      # Count models INSIDE the loop after variables are defined
      num_models=$(find results/${model}_${dataset}_${optimizer}/checkpoints/ -maxdepth 1 -type f -name "epoch*.pt" | sed 's/.*epoch\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
      echo "Found $num_models models for ${model}_${dataset}_${optimizer}"
      
      # Loop through epochs: 1, 10, 20, 30, ... up to num_models
      for i in 1 $(seq 10 10 "$num_models"); do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_${optimizer}_${dataset}_ep${i}
#SBATCH --account=ece556w26_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=output_${model}_${optimizer}_${dataset}.txt

/bin/hostname
pretrained_model="results/${model}_${dataset}_${optimizer}/checkpoints/epoch${i}.pt"

if [ ! -f "\$pretrained_model" ]; then
  echo "Skipping missing \$pretrained_model"
  exit 0
fi

if [ "${i}" -eq "${num_models}" ]; then
   python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/analysis/epoch${i} --pretrained_model="\$pretrained_model" --probe --max_train_data 10000
else
   python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/analysis/epoch${i} --pretrained_model="\$pretrained_model" --max_train_data 10000
fi
EOF
      done
    done
  done
done