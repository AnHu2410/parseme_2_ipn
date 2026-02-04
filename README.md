Code repository for the submission of Andrea Horbach’s research group to the PARSEME 2.0 Shared Task on Automatic Identification of Multiword Expressions.

Create and activate virtual environment based on requirements.txt.

Filtering (See Section 3): 
- Store sharedtask data in this directory (https://gitlab.com/parseme/sharedtask-data/-/tree/master/2.0). 
- Create directory, where filtered data can be stored: training_data/nonthinking/
- Run filtering.py

Create Thinking Data:
- Create directory thinking within training data (raining_data/thinking).
- Run create_thinking.py