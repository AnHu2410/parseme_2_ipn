"""
Corpus utilities for PARSEME 2.0 (CUPT) datasets.

This module defines a `Corpus` container that loads a single PARSEME 2.0 file
in CUPT format (CoNLL-U with an additional `PARSEME:MWE` column) and represents
it as a list of `Sentence` objects.

Each `Sentence` object contains:

self.sentence_string = cupt_obj.metadata.get("text")
        self.mwes = get_mwe_objects(self.cupt, self.sentence_string)
        self.preds_cupt = []
        self.preds = ""
- the sentence (e.g., `sentence_string`)
- gold MWEs occurring in the sentence (e.g., `mwes`)
- optional fields where predictions can be stored
- and methods to create thinking data for a given sentence.

In addition to reading/writing, the `Corpus` class provides helper methods for:
- storing model predictions aligned to sentences
- exporting data to a text-based "prediction" format
- exporting the "thinking format"
- basic corpus statistics (MWE token/type counts, hapax MWEs, and tag counts)

"""


from sentence import Sentence
from postprocess_predictions import *
import conllu
import parseme.cupt as cupt
from pathlib import Path
from helper_functions import read_prompt_from_file


class Corpus:
    def __init__(self, path: str):
        self.path = path  # path points to Parseme 2.0 data
        self.list_of_sentences = self.read_cupt_lines()
        self.prompt = self.read_prompt_from_file()

    def read_cupt_lines(self) -> list:
        # read a CUPT-formatted file
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.read()

            # collect all sentences and represent them as objects
            sentence_objects = []
            sentences = conllu.parse(lines)
            for sentence in sentences:
                sentence_obj = Sentence(sentence)
                sentence_objects.append(sentence_obj)
        
        return sentence_objects
    
    def store_preds(self, path_predictions):
        # store predictionss in sentence objects:
        with open(path_predictions, 'r', encoding='utf-8') as f:
            lines = f.read()
            preds = conllu.parse(lines)
            length_preds = len(preds)
            self.list_of_sentences = self.list_of_sentences[:length_preds]
            for ind, pred in enumerate(preds):
                self.list_of_sentences[ind].preds_cupt = pred
                self.list_of_sentences[ind].get_preds()
    
    def write_thinking_2_file(self, output_path):
        prompt = self.prompt
        for sentence in self.list_of_sentences:
            # generate thinking format for each sentence
            # separate sentence string and thinking 
            # part with +++++++, so it is easier to parse later
            sentence_str = sentence.sentence_string + " +++++++ "
            thinking = sentence.get_thinking_format()
            answer = prompt + sentence_str + thinking
            with open(output_path, "a") as o:
                o.write(answer)      
    
    def write_predictions_2_file(self, predictions_path, output_path):
        # writes predictions to cupt-formatted file:
        cupts = map_2_cupt(predictions_path, self.path)
        first_line = "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE"
        with open(output_path, "a") as f:
            f.write(first_line)
            for item in cupts:
                retrieved_cupt = item.serialize()
                f.write(retrieved_cupt)

    def write_cupt_2_txt(self, output_path):
        # write cupt data to txt file in prediction format:
        for sentence in self.list_of_sentences:
            mwe_str = sentence.cupt_2_prediction() + "\n"
            with open(output_path, "a") as f:
                f.write(mwe_str)
    
    
    def create_thinking_all_langs(self, directory_nonthinking, output_directory_thinking):
        # write thinking format for all languages to file for training data:
        langs = [
        "EGY", "EL", "FA", "FR", "HE", "JA", "KA",
        "LV", "NL", "PL", "PT", "RO", "SL", "SR", "SV", "UK",
        ]
        for lang in langs:
            corpus = Corpus(directory_nonthinking + "/" + lang + "/train_" + lang + ".cupt")
            output = output_directory_thinking + "/" + lang + "/train_" + lang + ".txt"
            corpus.write_thinking_2_file(directory_nonthinking, output, lang)

    # the following methods are used for analysing the corpora:         

    def count_MWE_types(self):
        # count MWE tokens, types and hapax MWEs:
        types = {}
        for sentence in self.list_of_sentences:
            for mwe in sentence.mwes:
                key = frozenset(mwe.lemmas)  # concatenated lemmas; limitation: 
                # some mwes can change the position of the lexemes; 
                #this is not taken into account here
                types[key] = types.get(key, 0) + 1
        count_hapax_mwes = 0
        for count in types.values():
            if count == 1:
                count_hapax_mwes += 1
        count_mwe_tokens = 0
        for mwe in types.values():
            count_mwe_tokens += mwe
        count_mwe_types = len(types)
        print("Count MWE tokens: ", count_mwe_tokens)
        print("Count MWE types: ", count_mwe_types)
        print("Count MWEs that occur only once: ", count_hapax_mwes)
        return count_mwe_tokens, count_mwe_types
    
    def count_MWE_tags(self):
        # count how many times each MWE tag occurs in the data for one language
        tags = {}
        for sentence in self.list_of_sentences:
            for mwe in sentence.mwes:
                key = mwe.mwe_tag
                tags[key] = tags.get(key, 0) + 1
        print("Count tags: ", tags)
        return tags

    def __str__(self):
        return self.path


# function for finding all tags in the training data of all languages:

def search_train_cupt_in_subdirs_for_tags(root_dir, skip_name="tools"):
    # loop over all training files in order to get complete tag set
    tags = []
    root = Path(root_dir) 
    if not root.is_dir(): 
        raise NotADirectoryError(root_dir)  

    for child in root.iterdir(): 
        if not child.is_dir(): 
            continue 
        if child.name == skip_name: 
            continue 
        f = child / "train.cupt" 
        if f.is_file(): 
            c = Corpus(f)
            for sentence in c.list_of_sentences:
                for mwe in sentence.mwes:
                    tags.append(mwe.mwe_tag)
    return set(tags)


# def write_cupt_w_preds_2_file(path, predictions_path, output_path):
#     c = Corpus(path)
#     preds= predictions_path
#     outp = output_path
#     c.write_predictions_2_file(preds, outp)
