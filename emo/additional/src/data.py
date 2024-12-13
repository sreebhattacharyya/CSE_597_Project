import os 

os.environ["HF_HOME"] = "/scratch/bbmr/sbhattacharyya1/models/"

import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset

class DataSetup(Dataset):
    def __init__(self, dataset_name, args, transform = None):
        self.dataset_name = dataset_name
        self.args = args

        self.labels = args.data[dataset_name]["labels"]
        self.label2idx = self.create_labels2idx()
        self.idx2label = self.create_idx2label()
        self.annotation_path = args.data[dataset_name]["annotation_path"]
        self.img_dir = args.data[dataset_name]["image_dir"]
        self.prompt_dir = args.data[dataset_name]["prompt_dir"]
        self.image_path_col = args.data[dataset_name]["image_path_col"]
        self.emotion_label_col = args.data[dataset_name]["emotion_label_col"]
        self.emotion_label_string = args.data[dataset_name]["emotion_label_string"]
        self.sentiment_label_col = args.data[dataset_name]["sentiment_label_col"]
        self.prompt = args.data[dataset_name]["prompt"]
        self.transform = transform

        self.annotation_df = pd.read_csv(self.annotation_path)

    def create_labels2idx(self):
        d = {}
        for i, label in enumerate(self.labels):
            d[label] = i
        return d

    def create_idx2label(self):
        d = {}
        for i, label in enumerate(self.labels):
            d[i] = label
        return d

    def __len__(self):
        return len(self.annotation_df)
    
    def __getitem__(self, idx): 
        row = self.annotation_df.iloc[idx]

        image_path = self.img_dir + row[self.emotion_label_col] + "/" + row[self.image_path_col] + ".jpg"
        # open raw image
        image = Image.open(self.img_dir + image_path)
        if self.transform: 
            image = self.transform(image)
        
        # read raw textual context
        # context = ""
        # with open(self.text_dir + row[self.image_path_col]) as f:
        #     context = f.read()
        # append additional prompt
        textual_prompt = ""
        try:
            with open(self.prompt) as f: 
                textual_prompt = f.read()
        except Exception as e: 
            print(f"Could not read prompt: {e}")
        
        if textual_prompt != "": 
            # fill in the emotion categories 
            emotion_categories_string = ', '.join(self.labels)
            textual_prompt = textual_prompt.replace('[emotion categories]', emotion_categories_string)

        # get emotion label
        emotion_label = row[self.emotion_label_col]
        if self.emotion_label_string:
            emotion_label = self.label2idx[emotion_label]
        
        # get sentiment label
        sentiment_label = row[self.sentiment_label_col]

        return image, textual_prompt, sentiment_label, emotion_label



