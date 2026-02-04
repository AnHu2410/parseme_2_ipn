"""Corpus object that can consist of any of the dev or train datasets from Parseme 2.0.
Object consists of a list of sentence objects. 
Each sentence object also contains a list of all MWEs 
occuring in the sentence (as objects).
It is also possible to count the number of MWE types, tokens, 
MWEs that occur only once and tags."""


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
        # store preds in sentence objects:
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


#  loop over all training files in order to get complete tag set
def search_train_cupt_in_subdirs_for_tags(root_dir, skip_name="tools"):
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


def create_thinking_all_langs():
    langs = [
    "EGY", "EL", "FA", "FR", "HE", "JA", "KA",
    "LV", "NL", "PL", "PT", "RO", "SL", "SR", "SV", "UK",
    ]
    for lang in langs:
        write_thinking_2_file(lang)


def get_cupt_data_from_predictions(language):
    c = Corpus("dev_data/dev_" + language + ".cupt")
    predictions_path = "dev_data/dev_p_Romance/dev_" + language + "_32.txt"
    output_path = "dev_data/dev_p_Romance/dev_" + language + "_32.cupt"
    c.write_predictions_2_file(predictions_path, output_path)




def get_error_ratios_tag(language):
    c = Corpus("dev_data/dev_" + language + ".cupt")
    predictions_path = "dev_data/dev_preds_thinking/dev_full_" + language + ".cupt"
    c.count_correctly_predicted_tags(predictions_path)


def write_thinking_2_file(language):
    corpus = Corpus("training_data/nonthinking/train_" + language + ".cupt")
    output = "training_data/thinking/train_" + language + ".txt"
    corpus.write_thinking_2_file(output)

def count_MWE_types_all_langs():
    language_dict = {}
    langs = [
    "EGY", "EL", "FA", "FR", "HE", "JA", "KA",
    "LV", "NL", "PL", "PT", "RO", "SL", "SR", "SV", "UK",
    ]
    for lang in langs:
        path = "/gxfs_home/cau/sunpn1133/sharedtask-data/2.0/subtask1/" + lang + "/train.cupt"
        c = Corpus(path)
        types, tokens = c.count_MWE_types()
        language_dict[lang] = types, tokens
    return language_dict





def create_cupt():
    p = "/gxfs_home/cau/sunpn1133/sharedtask-data/2.0/subtask1/SL/dev.cupt"
    #p = "intermediate/dev_SV.cupt"
    c = Corpus(p)
    preds= "results_paper/lfe/test_SL_sla2_cleaned.txt"
    outp = "results_paper/cupts/lfe_SL_sla2.cupt"
    c.write_predictions_2_file(preds, outp)

