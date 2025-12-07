from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
import os
import pandas as pd
from helper_functions import format_thinking, generate_conversation
import argparse
#from llmcompressor.transformers.finetune import oneshot


def parse_args():
    p = argparse.ArgumentParser(description="Quantize model and save it.")
    p.add_argument("-o", "--output", required=False, dest="output_data_path", help="Path where quantized models should be stored.")
    p.add_argument("-t", "--thinking", required=True, dest="thinking_data_path", help="Path to several thinking data files in txt format.")
    return p.parse_args()


def prepare_calibration_data(data_reasoning, tokenizer):
    dir_path = data_reasoning
    if not os.path.isdir(dir_path): 
        raise NotADirectoryError(dir_path)
    files = os.listdir(dir_path)
    td_sentences = []
    td_mwes = []
    for file in files:
        full_path = os.path.join(dir_path, file)
        if os.path.isfile(full_path): 
            with open(full_path, "r") as d:
                thinking_content = d.read()
                problems, answers = format_thinking(thinking_content)
                len_half = len(problems) // 2
                td_sentences += problems[:len_half]
                td_mwes += answers[:len_half]
    #dataset = {"sentences": td_sentences, "mwes": td_mwes}
    dataset = Dataset.from_dict({"sentences": td_sentences, "mwes": td_mwes})
    print("*** length Dataset: ", len(dataset))
    reasoning_conversations = tokenizer.apply_chat_template(list(dataset.map(generate_conversation, batched = True)["conversations"]), tokenize = False)
    print(type(reasoning_conversations))
    print("#### reasoning example: ", reasoning_conversations[0])
    print("#### reasoning length: ", len(reasoning_conversations))
    return reasoning_conversations


def quantize(model_dir, reasoning_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype="auto", device_map="auto")
    calibration_data = prepare_calibration_data(reasoning_dir, tokenizer)
    data = pd.Series(calibration_data)
    data.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    shuffled_dataset = dataset.select(range(500))
    # Configure the quantization algorithms
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=shuffled_dataset,
        recipe=recipe,
        max_seq_length=400,
        num_calibration_samples=200,
    )

    # Save the compressed model: 
    SAVE_DIR = "Qwen3-32B-INT8"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)



def main(thinking):
    model_dir = "../Qwen3-32B"
    quantize(model_dir, thinking)

if __name__ == "__main__":
	args = parse_args()
	main(args.thinking_data_path)
