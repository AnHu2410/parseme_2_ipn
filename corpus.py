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


class Corpus:
    def __init__(self, path: str):
        self.path = path  # path points to Parseme 2.0 data
        self.list_of_sentences = self.read_cupt_lines()

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
        with open(path_predictions, 'r', encoding='utf-8') as f:
            lines = f.read()
            preds = conllu.parse(lines)
            length_preds = len(preds)
            self.list_of_sentences = self.list_of_sentences[:length_preds]
            for ind, pred in enumerate(preds):
                self.list_of_sentences[ind].preds_cupt = pred
                self.list_of_sentences[ind].get_preds()
                self.list_of_sentences[ind].get_true_positives()
    
    def write_thinking_2_file(self, output_path):
        prompt = "You are a helpful system for identifying multiple-word expressions (MWEs). Identify all MWEs in the given sentence, and output their surface forms. Each sentence is a string of words delimited by'\\n'. An MWE is defined as a sequence that satisfies the following three conditions. 1. It consists of multiple words that are always realized by the same lexemes. The individual lexemes cannot be replaced by synonyms without distorting the meaning of the expression as a whole or violating language conventions. 2. It displays semantic, lexical, or syntactic idiomaticity. Semantic idiomaticity occurs when the meaning of an expression cannot be explicitly derived from its components. Lexical idiomaticity occurs when one or more components of an expression are not used as stand-alone words in standard English. Syntactic idiomaticity occurs when the grammar of an expression cannot be derived directly from that of its components. For example, semantically idiomatic MWEs include 'break up', the lexically idiomatic include 'to and fro', and syntactically idiomatic MWEs include 'long time no see'. 3. It is not a multi-word named entity, i.e., a specific name of a person, facility, etc.\n Remember that you can identify congruent MWEs across different languages. For example, you can identify the Romanian MWE 'pur și simplu' because you know the English MWE 'pure and simple'. Similarly, the Portuguese MWE 'ter lugar' is easy to identify because of the English MWE 'take place'. And the French MWE 'feu de circulation' is easy to identify, because it is almost congruent to the English MWE 'traffic lights'. Respond by providing all tokens of the MWE, and their indices. If no MWE occurs, output 'None'. If there are multiple MWEs, separate them by |, for example 'to and fro; 7,8,9 | break up; 12, 13'. Sentence: "
        print("Number of sentences: ", len(self.list_of_sentences))
        for sentence in self.list_of_sentences:
            
            sentence_str = sentence.sentence_string + " +++++++ "
            print(sentence_str)
            thinking = sentence.get_thinking_format()
            answer = prompt + sentence_str + thinking
            with open(output_path, "a") as o:
                o.write(answer)      
    
    def write_predictions_2_file(self, predictions_path, output_path):
        cupts = map_2_cupt(predictions_path, self.path)
        first_line = "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE"
        with open(output_path, "a") as f:
            f.write(first_line)
            for item in cupts:
                retrieved_cupt = item.serialize()
                f.write(retrieved_cupt)

    def write_cupt_2_txt(self, output_path):
        for sentence in self.list_of_sentences:
            mwe_str = sentence.cupt_2_prediction() + "\n"
            with open(output_path, "a") as f:
                f.write(mwe_str)

    # the following methods are used for analysing the corpora:         

    def count_MWE_types(self):
        # count how many mwe types occur only once
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
        tags = {}
        for sentence in self.list_of_sentences:
            for mwe in sentence.mwes:
                key = mwe.mwe_tag
                tags[key] = tags.get(key, 0) + 1
        print("Count tags: ", tags)
        return tags
    
    def count_correctly_predicted_sents(self, path_predictions):
        self.store_preds(path_predictions)
        correctly_predicted_sentences = 0
        for sentence in self.list_of_sentences:
            if sentence.predicted_correctly:
                correctly_predicted_sentences +=1
        print("Correctly predicted sentences: ", correctly_predicted_sentences)

    def count_correctly_predicted_tags(self, path_predictions):
        self.store_preds(path_predictions)

        # collect correctly predicted tags
        correctly_predicted_tags = {}
        for sentence in self.list_of_sentences:
            for mwe in sentence.mwes:
                if mwe.predicted_correctly:
                    if mwe.mwe_tag not in correctly_predicted_tags:
                        correctly_predicted_tags[mwe.mwe_tag] = 1
                    else:
                        correctly_predicted_tags[mwe.mwe_tag] += 1
        print("Correctly predicted tags: ", correctly_predicted_tags)

        # collect all tags in the gold data
        tags = {}
        for sentence in self.list_of_sentences:
            for mwe in sentence.mwes:
                key = mwe.mwe_tag
                tags[key] = tags.get(key, 0) + 1
        
        print("Number of tags in gold data: ", tags)
        
        # compute ratio between correctly predicted and all tags
        ratios = {}
        for tag in tags:
            if tag in correctly_predicted_tags:
                ratios[tag] = correctly_predicted_tags[tag] / tags[tag] 
        print("Ratios correctly predicted tags / all tags: ", ratios)
    
    def count_tps_noncontinuous(self, path_predictions):
        continuous_c = 0
        continuous_all = 0
        noncontinuous_c = 0
        noncontinuous_all = 0
        self.store_preds(path_predictions)
        for sentence in self.list_of_sentences:
            for mwe in sentence.mwes:

                if mwe.continuous_mwe:#
                    continuous_all += 1
                    if mwe.predicted_correctly:
                        continuous_c += 1
            
                else:
                    noncontinuous_all += 1
                    if mwe.predicted_correctly:
                        noncontinuous_c += 1

        print("continuous correctly predicted: ", continuous_c, "/", continuous_all, " ;noncontinuous correctly predicted",  noncontinuous_c, "/", noncontinuous_all,)

    def generate_thinking_data(self, language, output_path):
        if language == "PT":
            tags = {"ConjID": "", "DetID": "", "IAV": "", "IntjID": "", "IRV": "", "IVPC.full": "", "LVC.cause": ""}
        elif language == "FR":
            tags = {"AdjID": "", "AdpID": "", "AdvID": "", "LVC.full": "", "MVC": "", "NID": "", "PronID": ""}
        else: # RO
            tags = {"AV.IAV": "", "AV.VID": "", "NV.IAV": "", "NV.LVC.cause": "", "NV.LVC.full": "", "NV.VID": "", "VID": ""}
        for sent in self.list_of_sentences:
            for mwe in sent.mwes:
                tag = mwe.mwe_tag
                if tag in tags.keys():
                    if not tags[tag]:
                        tags[tag] = sent
                        break
        for sentence in  tags.values():
            with open(output_path, "a") as f:
                f.write(sentence.cupt.serialize() + "\n\n")

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


def create_thinking_all_langs():
    langs = [
    "EGY", "EL", "FA", "FR", "HE", "JA", "KA",
    "LV", "NL", "PL", "PT", "RO", "SL", "SR", "SV", "UK",
    ]
    for lang in langs:
        write_thinking_2_file(lang)


def create_cupt():
    p = "/gxfs_home/cau/sunpn1133/sharedtask-data/2.0/subtask1/SV/test.blind.cupt"
    c = Corpus(p)
    preds= "results/test_SV.txt"
    outp = "results/test_SV.cupt"
    c.write_predictions_2_file(preds, outp)


