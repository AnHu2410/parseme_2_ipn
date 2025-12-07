#!/bin/bash
#SBATCH --job-name=UK_inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=test_UK.out
#SBATCH --error=test_UK.err
#SBATCH --partition=gpu



# activate virtual environment
source /gxfs_work/cau/sunpn1133/new_env/bin/activate

# run python script
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i /gxfs_work/cau/sunpn1133/parseme_2_0/intermediate/test.blind_UK.cupt -o results/test_UK.txt -m lora_model_32_SLAVIC

jobinfo