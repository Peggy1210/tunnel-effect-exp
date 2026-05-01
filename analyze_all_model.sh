# # #!/bin/bash

# # models=("vgg19")
# # optimizers=("muon")
# # datasets=("cifar10")

# # for model in "${models[@]}"; do
# #   for optimizer in "${optimizers[@]}"; do
# #     for dataset in "${datasets[@]}"; do

# #       sbatch <<EOF
# # #!/bin/bash
# # #SBATCH --job-name=${model}_${optimizer}_${dataset}
# # #SBATCH --account=ece556w26_class
# # #SBATCH --partition=gpu
# # #SBATCH --gpus=1
# # #SBATCH --time=8:00:00
# # #SBATCH --nodes=1
# # #SBATCH --ntasks-per-node=1
# # #SBATCH --mem-per-cpu=64g
# # #SBATCH --mail-type=BEGIN,END
# # #SBATCH --output=output_${model}_${optimizer}_${dataset}.txt

# # /bin/hostname
# # num_models=$(find results/${model}_${dataset}_${optimizer}/checkpoints/ -maxdepth 1 -type f -name "epoch*.pt" | wc -l)

# # for i in 1 $(seq 10 10 "\$num_models"); do
# #   pretrained_model="results/\${model}_\${dataset}_\${optimizer}/checkpoints/epoch\${i}.pt"

# #   if [ ! -f "\$pretrained_model" ]; then
# #     echo "Skipping missing $pretrained_model"
# #     continue
# #   fi

# #   echo "Analyzing \$pretrained_model (model \$i of \$num_models)..."
# #   if [ "$i" -eq "\$num_models" ]; then
# #     python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/epoch${i} --pretrained_model="\$pretrained_model" --probe --max_train_data 500 --max_test_data 100 > /dev/null 2>&1
# #   else
# #     python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/epoch${i} --pretrained_model="\$pretrained_model" --max_train_data 500 --max_test_data 100 > /dev/null 2>&1
# #   fi
# # done
# # EOF

# #     done
# #   done
# # done


# model="resnet34"
# optimizer="fullbatch_gd"
# dataset="cifar10"

# num_models=$(find results/${model}_${dataset}_${optimizer}/checkpoints/ -maxdepth 1 -type f -name "epoch*.pt" | wc -l)

# for i in 1 $(seq 10 10 "$num_models"); do
#   pretrained_model="results/${model}_${dataset}_${optimizer}/checkpoints/epoch${i}.pt"

#   if [ ! -f "$pretrained_model" ]; then
#     echo "Skipping missing $pretrained_model"
#     continue
#   fi

#   echo "Analyzing $pretrained_model (model $i of $num_models)..."
#   if [ "$i" -eq "$num_models" ]; then
#     python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/epoch${i} --pretrained_model="$pretrained_model" --probe --max_train_data 1000 --max_test_data 5000
#   else
#     python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/epoch${i} --pretrained_model="$pretrained_model" --max_train_data 1000 --max_test_data 5000
#   fi
# done

#!/bin/bash

models=("resnet34")
optimizers=("adam")
datasets=("cifar10")

for model in "${models[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for dataset in "${datasets[@]}"; do
      # Count models INSIDE the loop after variables are defined
      num_models=$(find results/${model}_${dataset}_${optimizer}/checkpoints/ -maxdepth 1 -type f -name "epoch*.pt" | sed 's/.*epoch\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
      echo "Found $num_models models for ${model}_${dataset}_${optimizer}"
      
      # Loop through epochs: 1, 10, 20, 30, ... up to num_models
      for i in 200; do
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
   python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/epoch${i} --pretrained_model="\$pretrained_model" --probe --max_train_data 10000
else
   python main.py --model_name=${model} --optimizer=${optimizer} --dataset=${dataset} --num_workers=1 --experiment=${model}_${dataset}_${optimizer}/epoch${i} --pretrained_model="\$pretrained_model" --max_train_data 10000
fi
EOF
      done
    done
  done
done