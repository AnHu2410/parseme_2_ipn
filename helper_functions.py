"""Helper functions that are needed in multiple modules."""


import re
from pathlib import Path


def read_prompt_from_file(self) -> str:
        # read prompt from prompt.txt file
        with open('prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()


def shorten_output(output: str) -> str:

	"""Extract the model's user-facing answer.

	This function expects the model output to include a thinking marker
	(</think>) followed by an end marker (<|im_end|>). It returns the text
	between these markers. If the markers are missing, it returns an error
	string so callers can detect malformed outputs.
	"""
	# markers used by the current model/chat template
	think_marker = "</think>"
	end_marker = "<|im_end|>"
	think_pos = output.find(think_marker)
	end_pos = output.find(end_marker)
	# only return the substring if both markers are present in the right order
	if think_pos != -1 and end_pos != -1 and end_pos > think_pos:
		return output[think_pos + len(think_marker):end_pos].strip()
	else:
		# caller expects a string; return a clear error for later inspection
		return output[think_pos + len(think_marker):].strip()
	

def format_thinking(thinking_content):
    problems = []
    answers = []

    list_sents = thinking_content.split("*#*#*#*")
    for sent in list_sents:
        if sent.strip():
            problem_answer = sent.split("+++++++")
            problem = problem_answer[0].strip()
            problems.append(problem)
            answer = problem_answer[1].strip()
            answers.append(answer)

    return problems, answers


def generate_conversation(examples):
    problems  = examples["sentences"]
    solutions = examples["mwes"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    
    return { "conversations": conversations, }
