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
    collect_start = []
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
    token_list.append(current_token)
    return token_list

# def get_all_indices(pred, cupt_sentence):
#     all_indices = []

#     # split up multiple predictions
#     single_preds = pred.split("|")
#     if single_preds[0] == "None":
#         return all_indices

#     # loop over predictions and add them to cupt
#     for single_pred in single_preds:
#         if ";" not in single_pred:
#             return all_indices

#         # separate predicted tokens from predicted indices
#         tokens = single_pred.split(";")[0]
#         predicted_indices = single_pred.strip().split(";")[1]
#         predicted_indices = predicted_indices.split(",")

#         # get a list of all predicted tokens
#         token_list = split_w_apostrophes(tokens)

#         # First pass: collect candidate indices + a preliminary choice
#         #   per token: (candidate_indices, prelim_index)
#         per_token_data = []  # list of (candidates_list, prelim_index_or_None)

#         for token_ind, token in enumerate(token_list):
#             candidates = []

#             # collect all potential indices for this token in cupt
#             for cupt_tok in cupt_sentence:
#                 if norm(cupt_tok["form"]).strip() == norm(token).strip():
#                     if isinstance(cupt_tok["id"], tuple):
#                         for num in cupt_tok["id"]:
#                             if isinstance(num, int):
#                                 candidates.append(num)
#                     else:
#                         candidates.append(cupt_tok["id"])

#             prelim = None
#             if len(candidates) == 1:
#                 # only one option: use it
#                 prelim = candidates[0]
#             elif len(candidates) > 1:
#                 # more than one candidate: pick closest to predicted index
#                 min_diff = float("inf")
#                 best = None
#                 if len(predicted_indices) <= int(token_ind):
#                     # handles cases where there are fewer predicted indices
#                     # than MWE lexemes
#                     idx_for_pred = -1
#                 else:
#                     idx_for_pred = int(token_ind)

#                 predicted_index = int(predicted_indices[idx_for_pred])
#                 for cand in candidates:
#                     diff = abs(int(cand) - predicted_index)
#                     if diff < min_diff:
#                         min_diff = diff
#                         best = cand
#                 prelim = best

#             per_token_data.append((candidates, prelim))

#         # Second pass: refine tokens with multiple candidates using mean of others
#         # We compute mean of all other prelim indices (that are not None)
#         # and, for tokens with >1 candidate, choose the candidate closest to that mean.
#         final_indices = []

#         for i, (candidates, prelim) in enumerate(per_token_data):
#             if not candidates:
#                 # No candidates at all; keep whatever prelim is (likely None)
#                 final_indices.append(prelim)
#                 continue

#             if len(candidates) == 1:
#                 # Only one candidate → nothing to refine
#                 final_indices.append(prelim)
#                 continue

#             # Multiple candidates: try mean-based refinement
#             # Collect all other tokens' prelim indices (excluding this one)
#             other_indices = []
#             for j, (_, other_prelim) in enumerate(per_token_data):
#                 if j == i:
#                     continue
#                 if other_prelim is not None:
#                     other_indices.append(int(other_prelim))

#             if len(other_indices) == 0:
#                 # No information from others → fall back to prelim
#                 final_indices.append(prelim)
#                 continue

#             mean_other = sum(other_indices) / len(other_indices)

#             # pick candidate closest to mean_other
#             best = None
#             min_diff = float("inf")
#             for cand in candidates:
#                 diff = abs(int(cand) - mean_other)
#                 if diff < min_diff:
#                     min_diff = diff
#                     best = cand

#             final_indices.append(best)

#         # add this prediction's indices
#         all_indices.append(final_indices)

#     return all_indices


def get_all_indices(pred, cupt_sentence):
    all_indices = []
    # split up multiple predictions
    single_preds = pred.split("|")
    if single_preds[0] == "None":
        return all_indices
    # loop over predictions and add them to cupt
    for single_pred in single_preds:
        if not ";" in single_pred:
            return all_indices
        # separate predicted tokens from predicted indices
        tokens = single_pred.split(";")[0]
        predicted_indices = single_pred.strip().split(";")[1]
        predicted_indices = predicted_indices.split(",")
        
        # get a list of all predicted tokens and find their correct indices
        token_list = split_w_apostrophes(tokens)

        result_indices = []
        for token_ind, token in enumerate(token_list):
            potential_index = []
            # check for potential indices of tokens in cupt
            for cupt_tok in cupt_sentence:
                if norm(cupt_tok["form"]).strip() == norm(token).strip():
                    if isinstance(cupt_tok["id"], tuple):
                        for num in cupt_tok["id"]:
                            if isinstance(num, int):
                                potential_index.append(num)
                    else:
                        potential_index.append(cupt_tok["id"])

            # if more than one index is found for a token, 
            # use the one that is closest to the predicted index
            if len(potential_index)>1:
                min_diff = 10000
                result = ""
                for index_tokenid, tokenid in enumerate(potential_index):
                    if len(predicted_indices) <= int(token_ind):  # handles cases where there are 
                    #less predicted indices than mwe lexemes
                        token_ind = -1
                    predicted_index = predicted_indices[int(token_ind)]
                    diff = abs(int(tokenid)-int(predicted_index))
                    if diff < min_diff:
                        min_diff = diff
                        result = tokenid
                result_indices.append(result)
            # if there is only one index, use that one
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





    