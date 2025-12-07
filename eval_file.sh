#!/bin/bash

python3 $HOME/sharedtask-data/2.0/subtask1/tools/parseme_validate.py dev_data/dev_preds_nonthinking/dev_pred_FR.cupt --lang FR



python3 $HOME/sharedtask-data/2.0/subtask1/tools/parseme_evaluate.py --gold dev_data/dev_RO.cupt --pred /gxfs_home/cau/sunpn1133/results_Qwen14B/dev_data/dev_preds_thinking/dev_full_RO.cupt

