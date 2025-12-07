"""Sentence object to represent each sentence 
in the Parseme 2.0 dataset. It consists of the cupt object, a sentence string
and a list of all mwes that occur in this sentence, which are represented as objects."""


from mwe import MWE
import parseme.cupt as cupt
import conllu
import pandas as pd


def get_tag_dict():
    with open("tag_list.txt", "r") as t:
        lines = t.readlines()
        tag_dict = {}
        for line in lines: 
            if line.strip():
                line = line.split("\t")
                tag = line[0].strip()
                definition = line[1].strip()
                tag_dict[tag] = definition
    return tag_dict


def get_mwe_objects(cupt_f, sentence_string):
        mwe_objects = []
        mwes = cupt.retrieve_mwes(cupt_f)
        for mwe in mwes.values():
            tag = mwe.cat
            mwe_object = MWE(sentence_string, tag)
            span = mwe.span
            for token in cupt_f:
                if token["id"] in span:
                    mwe_object.tokens.append(token["form"])
                    mwe_object.lemmas.append(token["lemma"])
                    mwe_object.indices.append(token["id"])
            mwe_object.continuous_mwe = mwe_object.is_continuous()
            mwe_objects.append(mwe_object)
        return mwe_objects


class Sentence:
    def __init__(self, cupt_obj: object):
        self.cupt = cupt_obj
        self.sentence_string = cupt_obj.metadata.get("text")
        self.mwes = get_mwe_objects(self.cupt, self.sentence_string)
        self.preds_cupt = []
        self.preds = ""
        self.true_positives = 0
        self.predicted_correctly = False

    def get_thinking_format(self):
        tag_dict = get_tag_dict()
        starting_line = "<think>Okay, let's tackle this query. The user wants me to identify all multiple-word expressions (MWEs) in the given sentence. The sentence is split into words by newlines, so first I need to parse each line as a token. Let me list them out with their indices to keep track. The words are:\n"
        words_w_indices = ""
        for token in self.cupt:
            form = token["form"]
            ind = token["id"]
            if not isinstance(ind, int):
                ind = [str(element) for element in ind] 
                ind = "".join(ind) + "(a contracted form consisting of the two following lemmas)"
            line = str(ind) + " " + form + "\n"
            words_w_indices += line
        repeat_conditions = "Now we apply the MWE criteria. The first condition is that the expression consists of multiple words that are always realized by the same lexemes. The individual lexemes cannot be replaced by synonyms without distorting the meaning of the expression as a whole or violating language conventions. The second condition is that it displays semantic, lexical, or syntactic idiomaticity. Semantic idiomaticity occurs when the meaning of an expression cannot be explicitly derived from its components. Lexical idiomaticity occurs when one or more components of an expression are not used as stand-alone words in standard English. Syntactic idiomaticity occurs when the grammar of an expression cannot be derived directly from that of its components. For example, semantically idiomatic MWEs include 'break up', the lexically idiomatic include 'to and fro', and syntactically idiomatic MWEs include 'long time no see'. The third condition is that it is not a multi-word named entity. Looking at the words, let's check for possible MWEs. " 
        definitions = ""
        for mwe in self.mwes:
            mwe_tokens = " " + " ".join(mwe.tokens) + " "
            mwe_defintion = tag_dict[mwe.mwe_tag.strip()] + "\n"
            line = mwe_tokens + mwe_defintion
            definitions += line
        closing_line = "After considering all possibilities, my final answer is: "
        correct_answer = ""
        mwes = []
        for mwe in self.mwes:
            formatted_mwe = mwe.get_trainingdata_format()
            mwes.append(formatted_mwe)
        if len(mwes) == 0:
            pass
        elif len(mwes) == 1:
            correct_answer = correct_answer + mwes[0]
        else:
            for mwe in mwes:
                correct_answer = correct_answer + mwe + "|"
            correct_answer = correct_answer[:-1]
        closing_tag = "</think>"
        thinking = starting_line + words_w_indices + repeat_conditions + definitions + closing_line + correct_answer + closing_tag
        output = thinking + correct_answer + "*#*#*#*" + "\n\n"
        return output
        
    def get_preds(self):
        if self.preds_cupt:
            sentence_string = self.preds_cupt.metadata.get("text")
            self.preds = get_mwe_objects(self.preds_cupt, sentence_string)
        else:
            print("No predictions available for processing.")
    
    def get_true_positives(self):
        preds = [pred.lemmas for pred in self.preds]
        for mwe in self.mwes:
            if mwe.lemmas in preds:
                mwe.predicted_correctly = True
                self.true_positives += 1
        if len(self.mwes) == self.true_positives:
            self.predicted_correctly = True   

    def get_ratio_correctly_identified_tokens(self):
        identified_tokens = 0
        all_tokens = 0
        preds = []
        for pred_mwe in self.preds:
            for index in pred_mwe.indices:
                preds.append(index)

        for mwe in self.mwes:
            for index in mwe.indices:
                all_tokens += 1
                if index in preds:
                    identified_tokens += 1
        ratio = identified_tokens/all_tokens
        return ratio
    
    def cupt_2_prediction(self):
        sentence = self.sentence_string + "\t"
        mwes = []
        for mwe in self.mwes:
            formatted_mwe = mwe.get_trainingdata_format()
            mwes.append(formatted_mwe)
        if len(mwes) == 0:
            return sentence
        elif len(mwes) == 1:
            sentence += mwes[0]
            return sentence
        else:
            for mwe in mwes:
                sentence = sentence + mwe + "|"
            return sentence[:-1]
        
     
    def __str__(self):
        return self.sentence_string
        

  
