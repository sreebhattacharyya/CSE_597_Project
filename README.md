# CSE_597_Project

This repository is based on the code for the EmoVIT project (https://github.com/aimmemotion/EmoVIT/tree/main). To set the code up and run, follow the exact instructions available on the EmoVIT repository, which lists steps in detail to set up the environment and run the code. To run the additional experiments, check the main.py under emo/additional.

For training, testing, simply run the train.py or test.py in the LAVIS folder. For training with a subset, replace train.json with train_reduced.json. 
For running the additional experiments, run main.py under emo/additional, with options -m (mode = train/test), -d (dataset name = emoset). The model_type passed can also be changed to use a model different from EmoVIT.
