#!/bin/bash
#SBATCH --job-name=DEV_inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=12:30:00
#SBATCH --output=test_DEV.out
#SBATCH --error=test_DEV.err
#SBATCH --partition=gpu



# activate virtual environment
source /gxfs_work/cau/sunpn1133/new_env/bin/activate

# run python script
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_EL.cupt -o intermediate/results_dev/dev_EL.txt -m lora_model_32_GREEK
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_FA.cupt -o intermediate/results_dev/dev_FA.txt -m lora_model_32_FARSI
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_HE.cupt -o intermediate/results_dev/dev_HE.txt -m lora_model_32_HEBREW
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_JA.cupt -o intermediate/results_dev/dev_JA.txt -m lora_model_32_JAPANESE
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_KA.cupt -o intermediate/results_dev/dev_KA.txt -m lora_model_32nt_KA
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_EGY.cupt -o intermediate/results_dev/dev_EGY.txt -m lora_model_EGYPTIAN
python3 $WORK/parseme_2_0/inference_qwen_vllm.py -i  intermediate/dev_UK.cupt -o intermediate/results_dev/dev_UK.txt -m lora_model_32nt_SLAVIC



jobinfo