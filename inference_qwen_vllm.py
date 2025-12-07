from corpus import Corpus
from helper_functions import shorten_output
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import conllu
import parseme.cupt as cupt
import sys
import os
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# Module for running Qwen inference over Parseme-2.0 data.


def parse_args():
	p = argparse.ArgumentParser(description="Process unlabelled data with a model.")
	p.add_argument("-i", "--input", required=True, dest="unlabelled_data_path", help="Path to the unlabelled data file in cupt format.")
	p.add_argument("-o", "--output", required=True, dest="output_path", help="Path to the output file.")
	p.add_argument("-m", "--model", required=True, help="Model name or path.")
	return p.parse_args()


class QwenInferencer:
	def __init__(self, data_path, output_path, prompt, model_dir):
		self.model_dir = model_dir
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
		#self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, local_files_only=True, device_map="cuda", dtype=torch.float16)
		self.llm = LLM(model="../Qwen3-32B-INT8", enable_lora=True, max_model_len=5000, gpu_memory_utilization=0.85, enforce_eager=True)
		self.data_path = data_path	
		self.data = self.read_data_cupt()
		self.output_path = output_path
		self.prompt = prompt
		self.sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=5000)

	def read_data_cupt(self):
		data = []
		
		# Expect CUPT-format and turn each sentence into list of tokens
		with open(self.data_path, 'r', encoding='utf-8') as f:
			lines = f.read()
			sentences = conllu.parse(lines)
			for sentence in sentences:
				current_sent = ""
				for token in sentence:
					current_sent = current_sent + token["form"] + "\n"
				data.append(current_sent)
			
		return data

	def qwen_inference(self, sentence):

		prompt = self.prompt + str(sentence)
	
		messages = [{"role": "user", "content": prompt}]

		inputs = self.tokenizer.apply_chat_template(
				messages,
				add_generation_prompt=True,
				tokenize=False)
		#print("***Inputs: ", inputs)

		#outputs = self.model.generate(**inputs, max_new_tokens=5000)
		#answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
		outputs = self.llm.generate(inputs, self.sampling_params, lora_request=LoRARequest("lora_try_adapter", 1, self.model_dir))
		for output in outputs:
			#print("***output: ", output)
			output_ = output.outputs[0].text
			#print("**##**Output: ", output_)
			answer = shorten_output(output_)


			with open(self.output_path, "a") as f:
				# write a human-readable sentence (replace \n tokens with spaces)
				f.write(" ".join(sentence.split("\n")).strip() + "\t" + answer.strip() + "\n")
			
	def make_predictions(self):
		#self.model.eval()
		#with torch.no_grad():
		for sentence in self.data:
			self.qwen_inference(sentence)

def main(data_path, output_path, model):
	prompt = "You are a helpful system for identifying multiple-word expressions (MWEs). Identify all MWEs in the given sentence, and output their surface forms. Each sentence is a string of words delimited by'\\n'. An MWE is defined as a sequence that satisfies the following three conditions. 1. It consists of multiple words that are always realized by the same lexemes. The individual lexemes cannot be replaced by synonyms without distorting the meaning of the expression as a whole or violating language conventions. 2. It displays semantic, lexical, or syntactic idiomaticity. Semantic idiomaticity occurs when the meaning of an expression cannot be explicitly derived from its components. Lexical idiomaticity occurs when one or more components of an expression are not used as stand-alone words in standard English. Syntactic idiomaticity occurs when the grammar of an expression cannot be derived directly from that of its components. For example, semantically idiomatic MWEs include 'break up', the lexically idiomatic include 'to and fro', and syntactically idiomatic MWEs include 'long time no see'. 3. It is not a multi-word named entity, i.e., a specific name of a person, facility, etc.\n Remember that you can identify congruent MWEs across different languages. For example, you can identify the Romanian MWE 'pur și simplu' because you know the English MWE 'pure and simple'. Similarly, the Portuguese MWE 'ter lugar' is easy to identify because of the English MWE 'take place'. And the French MWE 'feu de circulation' is easy to identify, because it is almost congruent to the English MWE 'traffic lights'. Respond by providing all tokens of the MWE, and their indices. If no MWE occurs, output 'None'. If there are multiple MWEs, separate them by |, for example 'to and fro; 7,8,9 | break up; 12, 13'. Sentence: "
	q = QwenInferencer(data_path, output_path, prompt, model)
	q.make_predictions()

if __name__ == "__main__":
	args = parse_args()
	main(args.unlabelled_data_path, args.output_path, args.model)
	c = Corpus(args.unlabelled_data_path)
	c.write_predictions_2_file(args.output_path, args.output_path[:-3] + "cupt")