#!/bin/bash
#SBATCH --job-name=KA_finetuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=15:00:00
#SBATCH --output=test_train_KA.out
#SBATCH --error=test_train_KA.err
#SBATCH --partition=gpu
#SBATCH --constraint=H100


# activate virtual environment
source /gxfs_work/cau/sunpn1133/new_env/bin/activate

# run python script
python3 $WORK/parseme_2_0/finetune_qwen.py -n training_data/nonthinking/Georgian -t training_data/nonthinking/Georgian -o lora_model_32nt_KA
jobinfo