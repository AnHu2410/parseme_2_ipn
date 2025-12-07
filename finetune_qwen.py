from corpus import Corpus
import os, re
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, concatenate_datasets
import argparse
import pandas as pd
from helper_functions import format_thinking, generate_conversation


def parse_args():
    p = argparse.ArgumentParser(description="Finetune model on labelled data and save it.")
    p.add_argument("-n", "--nonthinking", required=True, dest="nonthinking_data_path", help="Path to several labelled data files in cupt format.")
    p.add_argument("-t", "--thinking", required=True, dest="thinking_data_path", help="Path to several thinking data files in txt format.")
    p.add_argument("-o", "--output_dir", required=True, dest="output_dir", help="Name of stored LoRa model.")
    p.add_argument("-s", "--eval", required=False, dest="eval", help="Eval dataset.")
    return p.parse_args()


class QwenFinetuner:
    def __init__(self, nonthinking, thinking, output_dir, eval):
        self.model, self.tokenizer = self.get_model_and_tokenizer()
        self.data_reasoning = thinking
        self.data_non_reasoning = nonthinking
        self.prompt = "You are a helpful system for identifying multiple-word expressions (MWEs). Identify all MWEs in the given sentence, and output their surface forms. Each sentence is a string of words delimited by'\\n'. An MWE is defined as a sequence that satisfies the following three conditions. 1. It consists of multiple words that are always realized by the same lexemes. The individual lexemes cannot be replaced by synonyms without distorting the meaning of the expression as a whole or violating language conventions. 2. It displays semantic, lexical, or syntactic idiomaticity. Semantic idiomaticity occurs when the meaning of an expression cannot be explicitly derived from its components. Lexical idiomaticity occurs when one or more components of an expression are not used as stand-alone words in standard English. Syntactic idiomaticity occurs when the grammar of an expression cannot be derived directly from that of its components. For example, semantically idiomatic MWEs include 'break up', the lexically idiomatic include 'to and fro', and syntactically idiomatic MWEs include 'long time no see'. 3. It is not a multi-word named entity, i.e., a specific name of a person, facility, etc.\n Remember that you can identify congruent MWEs across different languages. For example, you can identify the Romanian MWE 'pur și simplu' because you know the English MWE 'pure and simple'. Similarly, the Portuguese MWE 'ter lugar' is easy to identify because of the English MWE 'take place'. And the French MWE 'feu de circulation' is easy to identify, because it is almost congruent to the English MWE 'traffic lights'. Respond by providing all tokens of the MWE, and their indices. If no MWE occurs, output 'None'. If there are multiple MWEs, separate them by |, for example 'to and fro; 7,8,9 | break up; 12, 13'. Sentence: "
        self.output_dir = output_dir
        self.eval = eval
    
    def get_model_and_tokenizer(self):
        # get model and tokenizer:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "../Qwen3-32B",
            max_seq_length = 1000,   # Context length - can be longer, but uses more memory
            load_in_4bit = True,     # 4bit uses much less memory
            load_in_8bit = False,    # A bit more accurate, uses 2x memory
            full_finetuning = False, # We have full finetuning now!
            # token = "hf_...",      # use one if using gated models
        )

        # add LoRa-adapters to model:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )

        return model, tokenizer

    def prepare_non_reasoning_training_data(self):
        dir_path = self.data_non_reasoning
        if not os.path.isdir(dir_path): 
            raise NotADirectoryError(dir_path)
        files = os.listdir(dir_path)
        td_sentences = []
        td_mwes = []
        for file in files:
            full_path = os.path.join(dir_path, file)
            if os.path.isfile(full_path): 
                corpus = Corpus(full_path)
                list_of_sentences = corpus.list_of_sentences
                #len_half = len(list_of_sentences)
                #list_of_sentences = list_of_sentences[len_half:]
                for sentence in list_of_sentences:
                    tokens = []
                    for token in sentence.cupt:
                        tokens.append(token["form"])
                    sent = self.prompt + "\n".join(tokens)
                    td_sentences.append(sent)
                    sentence_mwes = sentence.cupt_2_prediction()
                    mwe = sentence_mwes.split("\t")[1]
                    td_mwes.append(mwe)
        #data_dict = {"sentences": td_sentences, "mwes": td_mwes}
        dataset = Dataset.from_dict({"sentences": td_sentences, "mwes": td_mwes})
        nonreasoning_conversations = self.tokenizer.apply_chat_template(list(dataset.map(generate_conversation, batched = True)["conversations"]), tokenize = False)
        
        print(nonreasoning_conversations[0])
        print(nonreasoning_conversations[3])
        print(nonreasoning_conversations[10])
        
        return nonreasoning_conversations

    def prepare_reasoning_training_data(self):
        dir_path = self.data_reasoning
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
                    #len_half = len(problems) // 2
                    td_sentences += problems # [:len_half]
                    td_mwes += answers # [:len_half]
        #dataset = {"sentences": td_sentences, "mwes": td_mwes}
        dataset = Dataset.from_dict({"sentences": td_sentences, "mwes": td_mwes})
        print("*** length Dataset: ", len(dataset))
        reasoning_conversations = self.tokenizer.apply_chat_template(list(dataset.map(generate_conversation, batched = True)["conversations"]), tokenize = False)
        print(type(reasoning_conversations))
        print("#### reasoning example: ", reasoning_conversations[0])
        print("#### reasoning length: ", len(reasoning_conversations))
        return reasoning_conversations
    
    def combine_data(self, mode="reasoning"):
        data_reasoning = self.prepare_non_reasoning_training_data()

        if mode == "reasoning":
            # Only reasoning data
            data = pd.Series(data_reasoning)
        else:
            # Combine reasoning + non‑reasoning data
            data_nonreasoning = self.prepare_non_reasoning_training_data()
            data = pd.concat([
                pd.Series(data_reasoning),
                pd.Series(data_nonreasoning),
            ])

        data.name = "text"
        combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
        combined_dataset = combined_dataset.shuffle(seed=3407)
        print("###############Length combined dataset: ", len(combined_dataset))
        return combined_dataset
    
    def get_eval_data(self):
        dir_path = self.eval
        if not os.path.isdir(dir_path): 
            raise NotADirectoryError(dir_path)
        files = os.listdir(dir_path)
        td_sentences = []
        td_mwes = []
        for file in files:
            full_path = os.path.join(dir_path, file)
            if os.path.isfile(full_path): 
                corpus = Corpus(full_path)
                list_of_sentences = corpus.list_of_sentences
                for sentence in list_of_sentences:
                    tokens = []
                    for token in sentence.cupt:
                        tokens.append(token["form"])
                    sent = self.prompt + "\n".join(tokens)
                    td_sentences.append(sent)
                    sentence_mwes = sentence.cupt_2_prediction()
                    mwe = sentence_mwes.split("\t")[1]
                    td_mwes.append(mwe)
        data_dict = {"sentences": td_sentences, "mwes": td_mwes}
        dataset = Dataset.from_dict(data_dict).shuffle(seed=42)
        nonreasoning_conversations = self.tokenizer.apply_chat_template(list(dataset.map(generate_conversation, batched = True)["conversations"]), tokenize = False)
        
        print("eval: ", nonreasoning_conversations[0])
        
        return nonreasoning_conversations
    
    def train_and_save_model(self):
        combined_dataset = self.combine_data()
        print("made it")
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = combined_dataset,
            eval_dataset = None, # Can set up evaluation
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 8, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 2,
                # max_steps = 30,
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use TrackIO/WandB etc,
            ),
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


        self.model.save_pretrained(self.output_dir)  # Local saving
        self.tokenizer.save_pretrained(self.output_dir)


def main(nonthinking, thinking, output_dir, eval):
    q = QwenFinetuner(nonthinking, thinking, output_dir, eval)
    q.train_and_save_model()


if __name__ == "__main__":
	args = parse_args()
	main(args.nonthinking_data_path, args.thinking_data_path, args.output_dir, args.eval)
