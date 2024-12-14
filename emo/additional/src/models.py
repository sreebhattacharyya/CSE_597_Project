import os 

os.environ["HF_HOME"] = "/scratch/bbmr/sbhattacharyya1/models/"

import pandas 
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess

class VLM(nn.Module):
    def __init__(self, device, vlm_name):
        super(VLM, self).__init__()
        self.vlm_name = vlm_name
        self.model, self.vis_processors, self.text_processors = None, None, None
        self.device = device
        if self.vlm_name == "blip":
            self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    
    def forward(self, raw_image, prompt):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        text = self.text_processors["eval"](prompt)
        sample = {"image": image, "text_input": [text]}

        features = self.model.extract_features(sample)
        return features

class RewardModel(nn.Module):
    def __init__(self, device, input_dim, output_dim): 
        super(RewardModel, self).__init__()
        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_dim)
        )
        self.predict_layer = nn.Linear(output_dim, 2)


    def forward(self, input):

        reward_state = self.layers(input)
        logits = self.predict_layer(reward_state)

        return reward_state, logits

class PunishModel(nn.Module):
    def __init__(self, device, input_dim, output_dim): 
        super(PunishModel, self).__init__()
        self.device = device

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_dim)
        )
        self.predict_layer = nn.Linear(output_dim, 2)


    def forward(self, input):

        punish_state = self.layers(input)
        logits = self.predict_layer(punish_state)

        return punish_state, logits

class RewardPunishModel(nn.Module):
    def __init__(self, device, lm_name, num_classes):
        super(RewardPunishModel, self).__init__()
        self.vlm = VLM(device, lm_name)
        
        self.reward_embedding = nn.Parameter(torch.randn(768, 768), requires_grad=True)
        self.punish_embedding = nn.Parameter(torch.randn(768, 768), requires_grad=True)
        
        self.reward_model = RewardModel(device, 768, 128)
        self.punish_model = PunishModel(device, 768, 128)

        self.fuse_attn = nn.MultiheadAttention(128, 2, batch_first=True) # embed_dim, num_heads; embed_dim % num_heads must be 0.
        
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, text):
        multimodal_features = self.vlm(image, text)

        reward = torch.matmul(multimodal_features, self.reward_embedding)
        punish = torch.matmul(multimodal_features, self.punish_embedding)

        reward_state, reward_logit = self.reward_model(reward)
        punish_state, punish_logit = self.punish_model(punish)

        fused_features = torch.cat([reward_state, punish_state], dim=0)

        attn_output, _ = self.fuse_attn(fused_features, fused_features, fused_features)

        pooled_attn_output = attn_output.mean(dim=0)

        final_pred = self.final(pooled_attn_output)

        return final_pred, reward_logit, punish_logit

