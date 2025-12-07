import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from corpus import Corpus

c = Corpus("/gxfs_home/cau/sunpn1133/results_Qwen14B/dev_data/dev_PT.cupt")
c.store_preds("/gxfs_home/cau/sunpn1133/results_Qwen14B/dev_data/dev_preds_thinking/dev_full_PT.cupt")


d = Corpus("/gxfs_home/cau/sunpn1133/results_Qwen14B/dev_data/dev_PT.cupt")
d.store_preds("/gxfs_home/cau/sunpn1133/results_Qwen14B/dev_data/dev_preds_nonthinking/dev_full_PT.cupt")

def get_evaluated_preds(corpus_w_preds):
    preds = []
    for sentence in corpus_w_preds.list_of_sentences:
        for mwe in sentence.mwes:
            if mwe.predicted_correctly:
                preds.append(1)
            else:
                preds.append(0)
    return preds

def calcualte_mcnemar(corpus_w_preds1, corpus_w_preds2):
    preds1 = get_evaluated_preds(corpus_w_preds1)
    preds2 = get_evaluated_preds(corpus_w_preds2)

    y1 = np.array(preds1)  # model 1 correctness, 0/1
    y2 = np.array(preds2)  # model 2 correctness, 0/1

    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")

        # Contingency counts
    a = np.sum((y1 == 1) & (y2 == 1))  # both correct
    b = np.sum((y1 == 0) & (y2 == 1))  # model1 wrong, model2 correct
    c = np.sum((y1 == 1) & (y2 == 0))  # model1 correct, model2 wrong
    d = np.sum((y1 == 0) & (y2 == 0))  # both wrong

    print("Contingency table:")
    print(f"          model2 correct   model2 wrong")
    print(f"model1 correct     {a:5d}           {c:5d}")
    print(f"model1 wrong       {b:5d}           {d:5d}")
    table = np.array([[a, b],
                  [c, d]])

    # exact=True recommended when b + c is small
    result = mcnemar(table, exact=True, correction=False)

    print("statistic:", result.statistic)
    print("p-value:", result.pvalue)

calcualte_mcnemar(c, d)