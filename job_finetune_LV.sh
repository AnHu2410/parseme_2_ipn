#!/bin/bash
#SBATCH --job-name=LV_finetuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=15:00:00
#SBATCH --output=test_train_LV.out
#SBATCH --error=test_train_LV.err
#SBATCH --partition=gpu
#SBATCH --constraint=H100


# activate virtual environment
source /gxfs_work/cau/sunpn1133/new_env/bin/activate

# run python script
python3 $WORK/parseme_2_0/finetune_qwen.py -n training_data/nonthinking/Latvian -t training_data/nonthinking/Latvian -o lora_model_32nt_LATVIAN
jobinfo