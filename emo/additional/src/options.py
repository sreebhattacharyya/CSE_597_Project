import argparse 
import os

class DefaultArgs():
    def __init__(self, exp_id):
        self.exp_id = exp_id

        self.data = {
            "emoset": {
                "labels": ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"],
                "annotation_path": "/work/hdd/bbmr/sbhattacharyya1/projects-src/llm-eval/Emo-Set/data/annotations_full.csv",
                "image_dir": "/work/hdd/bbmr/sbhattacharyya1/projects-src/llm-eval/Emo-Set/data/image/",
                "image_path_col": "image_id",
                "emotion_label_col": "emotion",
                "emotion_label_string": True,
                "sentiment_label_col": "sentiment",
                "prompt": "/work/hdd/bbmr/sbhattacharyya1/projects-src/neuro-inspired/prompts/general.txt",
                # "text_dir": 
            }
        }

        self.max_epochs = 1 # change this according to training needs
        self.alpha = 0.6
        self.beta = 0.2 
        self.gamma = 0.2