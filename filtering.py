"""This class performs filtering of training data: one sentence per MWE type."""

import random
from corpus import Corpus


class FilterMweTypes:
    def __init__(self, corpus):
        self.corpus = corpus
        self.mwe_types = self.get_mwe_types()
        self.one_per_type = self.get_one_MWE_per_type()
    
    def get_mwe_types(self):
        # create dictionary with mwe types (concatenated lemmas) as keys and
        # a list of all sentences that contain this mwe type as values
        mwe_types = {}
        empty_sents = []
        for sentence in self.corpus.list_of_sentences:
            if not sentence.mwes:
                empty_sents.append(sentence)
            else:
                for mwe in sentence.mwes:
                    type_as_set = frozenset(mwe.lemmas)
                    if type_as_set not in mwe_types.keys():
                        mwe_types[type_as_set] = [sentence]
                    else:
                        mwe_types[type_as_set] += [sentence]
        return mwe_types, empty_sents
    
    def get_one_sentene_per_type(self):
        sentence_list = []
        mwe_types, empty_sents = self.get_mwe_types()
        for mwe_type, sentences in mwe_types.items():
            if sentences[0] not in sentence_list:
                sentence_list.append(sentences[0])
        count_added_empty = 0
        for sentence in empty_sents:
            if sentence not in sentence_list:
                sentence_list.append(sentence)
                count_added_empty += 1
            if count_added_empty == 20:
                break

        return sentence_list
    
    def get_two_sentences_per_type(self):
        sentence_list = []
        mwe_types, empty_sents = self.get_mwe_types()
        for mwe_type, sentences in mwe_types.items():
            k = min(2, len(sentences))  # pick up to 2 items, but not more than exist
            picked = random.sample(sentences, k)
            for picked_sentence in picked:
                if picked_sentence not in sentence_list:
                    sentence_list.append(picked_sentence)
        count_added_empty = 0
        for sentence in empty_sents:
            if sentence not in sentence_list:
                sentence_list.append(sentence)
                count_added_empty += 1
            if count_added_empty == 20:
                break

        return sentence_list

    def count_mwe_types(self):
        # make sure all types/tokens that are contained in data
        # are included
        print(len(self.mwe_types))
    
    def get_one_MWE_per_type(self):
        seen = set()
        collected = []
        for sentence in self.corpus.list_of_sentences:
            types = {str(m) for m in sentence.mwes}
            if not types:
                continue
            # Only accept the sentence if all its types are unseen
            if types.isdisjoint(seen):
                collected.append(sentence)
                seen.update(types)
        return collected

    def write_one_MWE_per_type_2_file(self, lang):
        # write filtered data to file
        with open("training_data/nonthinking/train_" + lang + ".cupt", "a") as f:
            print("Length filtered sentences " + lang + " :", len(self.get_one_sentene_per_type()))
            for sentence in self.get_one_sentene_per_type():
                f.write(sentence.cupt.serialize() + "\n\n")

def write_filtered_training_data_2_file():
    languages = [
    "EGY", "EL", "FA", "FR", "HE", "JA", "KA",
    "LV", "NL", "PL", "PT", "RO", "SL", "SR", "SV", "UK",
    ]
    for lang in languages:
        path = "/gxfs_home/cau/sunpn1133/sharedtask-data/2.0/subtask1/" + lang + "/train.cupt"
        c = Corpus(path)
        f = FilterMweTypes(c)
        f.write_one_MWE_per_type_2_file(lang)


