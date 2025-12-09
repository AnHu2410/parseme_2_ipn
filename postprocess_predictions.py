"""Functions for postprocessing the LLM predictions."""


from helper_functions import shorten_output
import conllu
import parseme.cupt as cupt
from parseme.cupt import MWE
import re
import unicodedata as ud


def norm(s: str) -> str:
    return ud.normalize("NFC", s)


def split_w_apostrophes(tokens):
    token_list = []
    current_token = ""
    for char in tokens:
        if char == " " and current_token:
            token_list.append(current_token)
            current_token = ""
        elif char == "'" and current_token:
            current_token += char
            token_list.append(current_token)
            current_token = ""
        else:
            current_token += char
    if current_token:
        token_list.append(current_token)
    return token_list


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # DP table
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)

    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def get_all_indices(pred, cupt_sentence):
    all_indices = []
    single_preds = pred.split("|")
    if single_preds[0].strip() == "None":
        return all_indices

    for single_pred in single_preds:
        single_pred = single_pred.strip()
        if ";" not in single_pred:
            continue

        tokens = single_pred.split(";")[0]
        predicted_indices = single_pred.strip().split(";")[1]
        if "-" in predicted_indices:
            predicted_indices = predicted_indices.replace('-', ',')
        predicted_indices = [x.strip() for x in predicted_indices.split(",") if x.strip()]

        token_list = split_w_apostrophes(tokens)

        result_indices = []
        for token_ind, token in enumerate(token_list):
            potential_index = []

            matched = False
            for cupt_tok in cupt_sentence:
                if norm(cupt_tok["form"]).strip() == norm(token).strip():
                    matched = True
                    if isinstance(cupt_tok["id"], tuple):
                        for num in cupt_tok["id"]:
                            if isinstance(num, int):
                                potential_index.append(num)
                    else:
                        potential_index.append(cupt_tok["id"])

            if not matched:
                best_dist = float("inf")
                best_form = None
                best_indices = []

                for cupt_tok in cupt_sentence:
                    cupt_form = norm(cupt_tok["form"]).strip()
                    dist = levenshtein(norm(token).strip(), cupt_form)
                    if dist < best_dist:
                        best_dist = dist
                        best_form = cupt_form
                        best_indices = []
                        if isinstance(cupt_tok["id"], tuple):
                            for num in cupt_tok["id"]:
                                if isinstance(num, int):
                                    best_indices.append(num)
                        else:
                            best_indices.append(cupt_tok["id"])
                    elif dist == best_dist:
                        if isinstance(cupt_tok["id"], tuple):
                            for num in cupt_tok["id"]:
                                if isinstance(num, int):
                                    best_indices.append(num)
                        else:
                            best_indices.append(cupt_tok["id"])

                token = best_form
                potential_index = best_indices

            if len(potential_index) > 1:
                min_diff = 10000
                result = ""
                for tokenid in potential_index:
                    if len(predicted_indices) <= int(token_ind):
                        idx_for_pred = -1
                    else:
                        idx_for_pred = token_ind
                    predicted_index = predicted_indices[int(idx_for_pred)]
                    diff = abs(int(tokenid) - int(predicted_index))
                    if diff < min_diff:
                        min_diff = diff
                        result = tokenid
                result_indices.append(result)

            elif len(potential_index) == 1:
                result = potential_index[0]
                result_indices.append(result)

        all_indices.append(result_indices)

    return all_indices


def map_2_cupt(predictions:str, cupt_file:str):
    with open(predictions, "r") as p:  # raw predictions in format 'all in all; 7, 8, 9 | rain cats and dogs; 13, 14, 15, 16'
        predictions_list = p.readlines()
        len_prediction = len(predictions_list)
    
    with open(cupt_file, "r") as c: # original cupt formatted file
        cupt_f = c.read()
        cupt_sentences = conllu.parse(cupt_f)[:len_prediction]

    # clear mwes from original cupt file and make list of cupt sentences
    
    for cupt_sent in cupt_sentences:
        cupt.clear_mwes(cupt_sent)
    
    # loop over predictions and add them to cupt sentences
    for pred_ind, prediction in enumerate(predictions_list):
        print("####PRED####", prediction)
        sent_and_pred = prediction.split("\t")
        print("list#####", sent_and_pred)
        sent = sent_and_pred[0]
        pred = sent_and_pred[1]
        
        # digit → letter  (e.g. 5ა → 5|ა)
        pred = re.sub(r'(\d)\s*([^\W\d_])', r'\1|\2', pred, flags=re.UNICODE)
        print("#####pred_filtered", pred)
    
        current_cupt = cupt_sentences[pred_ind]
        all_indices = get_all_indices(pred, current_cupt)
        #print(all_indices)
        if all_indices == []:
            pass
        
        else:
            for i, index in enumerate(all_indices):
                if all_indices[i]:
                        cupt.add_mwe(current_cupt, mwe_id=i+1, mwe=MWE('VID', set(all_indices[i])))
    
    return cupt_sentences




def clean_files():
    preds = "results/test_EGY.txt"
    with open(preds, "r") as pr:
        text = pr.read()
        # remove all \n before lines where there is no \t. \t separates 
        # sentences from predictions; if there is no \t, it is a part 
        # of a prediction that was wrongly assigned a new line:
        text = re.sub(r'\n(?=[^\t\r\n]*(?:\n|$))', '', text)
        # Remove thinking that went on too long and 
        # did not produce an answer within the given token limit:
        text = re.sub(r"Okay, let's tackle this query[^\n]*", "None", text)

        with open("cleaned_files/EGY_cleaned.txt", "w") as o:
            o.write(text)

clean_files()