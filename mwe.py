"""This MWE oject contains the MWE category (e.g. "AdvID") which we call tag (in order to 
not confuse it with mwe types, e.g. "pig in a poke"), the sentence 
string of the sentence it was found in, as well as lists of all its tokens and lemmas."""
    


class MWE:
    def __init__(self, sentence_string, tag):
        self.sentence = sentence_string
        self.mwe_tag = tag
        self.mwe_thinking = ""
        self.tokens = []
        self.lemmas = []
        self.indices = []
        self.predicted_correctly = False
        self.continuous_mwe = False
    
    def get_trainingdata_format(self):
        indices_str = [str(index) for index in self.indices]
        return " ".join(self.tokens) + "; " + ",".join(indices_str)
    
    def is_continuous(self):
        indices = [int(index) for index in self.indices]
        sorted_indices = sorted(indices)
        return all(b - a == 1 for a, b in zip(sorted_indices, sorted_indices[1:]))
    
    def __str__(self):
        return " ".join(self.lemmas)
