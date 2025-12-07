#!/bin/bash
#SBATCH --job-name=KA_inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=44:00:00
#SBATCH --output=test_KA.out
#SBATCH --error=test_KA.err
#SBATCH --partition=gpu



# activate virtual environment
source /gxfs_work/cau/sunpn1133/new_env/bin/activate

# run python script
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  /gxfs_home/cau/sunpn1133/sharedtask-data/2.0/subtask1/KA/test.blind.cupt -o results/test_KA.txt -m lora_model_32nt_KA

jobinfo