"""Requires a directory within training_data called thinking (training_data/thinking).
Here the thinking data is stored."""


from corpus import Corpus


def write_thinking_2_file(language):
    corpus = Corpus("training_data/nonthinking/train_" + language + ".cupt")
    output = "training_data/thinking/train_" + language + ".txt"
    corpus.write_thinking_2_file(output)

def create_thinking_all_langs():
    langs = [
    "EGY", "EL", "FA", "FR", "HE", "JA", "KA",
    "LV", "NL", "PL", "PT", "RO", "SL", "SR", "SV", "UK",
    ]
    for lang in langs:
        write_thinking_2_file(lang)

if __name__ == "__main__":
    create_thinking_all_langs()