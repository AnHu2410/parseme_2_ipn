"""This class performs filtering of training data: it returns 
one sentence per MWE type, plus
up to 20 additional sentences without MWEs, avoiding duplicates. 

It requires sharedtask-data to be in the same directory as this file.

It requires a directory structure like this: training_data/nonthinking/
There, the output is stored."""


from corpus import Corpus


class FilterMweTypes:
    def __init__(self, corpus):
        self.corpus = corpus
        self.mwe_types = self.get_mwe_types()
        self.one_per_type = self.get_one_MWE_per_type()
    
    def get_mwe_types(self):
        """Extract MWE types from the corpus and organize by type.
        
        MWE types are defined as unique combinations of lemmas (represented as
        frozensets to be order-independent). Sentences are grouped by the MWE
        types they contain.
        
        Returns:
            tuple: A tuple containing:
                - mwe_types (dict): Dictionary with frozenset(lemmas) as keys and
                  lists of sentences as values.
                - empty_sents (list): List of sentences that contain no MWEs.
        """
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
        # for each MWE type, return one sentence that contains it, avoiding duplicates
        sentence_list = []
        mwe_types, empty_sents = self.get_mwe_types()
        for mwe_type, sentences in mwe_types.items():
            if sentences[0] not in sentence_list:
                sentence_list.append(sentences[0])
        # add up to 20 sentences without MWEs, avoiding duplicates
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
        path = "sharedtask-data/2.0/subtask1/" + lang + "/train.cupt"  # requires sharedtask-data
        c = Corpus(path)
        f = FilterMweTypes(c)
        f.write_one_MWE_per_type_2_file(lang)


if __name__ == "__main__":
    write_filtered_training_data_2_file()
