import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import time
import json

def Trainer(): 
    def __init__(self, train_data, val_data, test_data, args, model, device):
        self.train_data = train_data
        self.test_data = test_data 
        self.val_data = val_data
        self.args = args
        self.model = model
        self.device = device
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_accuracy(self, o_emotion, gt_emotion, o_reward_sent, o_punish_sent, gt_sent):
        
        emotion_acc = accuracy_score(gt_emotion, o_emotion)
        reward_model_accuracy = accuracy_score(gt_sent, o_reward_sent)
        gt_sent_punish = [1 - x for x in gt_sent] # the labels are flipped for the punish model
        punish_model_accuracy = accuracy_score(gt_sent_punish, o_punish_sent)

        print(f"Accuracy for Overall Emotion Classification = {emotion_acc}")
        print(f"Accuracy of the Reward Model = {reward_model_accuracy}")
        print(f"Accuracy of the Punish Model = {punish_model_accuracy}")

    
    def get_f1_score(self, o_emotion, gt_emotion, o_reward_sent, o_punish_sent, gt_sent):

        emotion_f1 = f1_score(gt_emotion, o_emotion, average='weighted')
        reward_model_f1 = f1_score(gt_sent, o_reward_sent, average='weighted')
        gt_sent_punish = [1 - x for x in gt_sent]
        punish_model_f1 = f1_score(gt_sent_punish, o_punish_sent)

        print(f"F1 Score for Overall Emotion Classification = {emotion_f1}")
        print(f"F1 of the Reward Model = {reward_model_f1}")
        print(f"F1 of the Punish Model = {punish_model_f1}")

    
    def train(self):
        print(f"Training model using device = {self.device}")
        since = time.time()

        for epoch in range(self.args.max_epochs): 
            print(f"Epoch {epoch + 1} / {self.args.max_epochs}:")
            for image, text, sentiment_label, emotion_label in self.train_data: 
                pred, reward_logit, punish_logit = self.model(image, text)

                combined_loss = self.args.alpha * self.criterion_ce(pred, emotion_label) + self.args.beta * self.criterion_bce(reward_logit, sentiment_label) + self.args.gamma * self.criterion_bce(punish_logit, (1-sentiment_label))
                combined_loss.backward()
                
                self.optimizer.step()
            
            if epoch % 10 == 0: 
                print(f"Current Loss = {combined_loss}")
                self.test()
            print()
    
    def test(self):
        
        output_emotion = []
        ground_truth_emotion = []

        output_reward_model_sent = []
        output_punish_model_sent = []

        ground_truth_sent = []

        for image, text, sentiment_label, emotion_label in self.train_data: 
            pred, reward_logit, punish_logit = self.model(image, text)

            _, pred_emotions = torch.max(pred, 1) 
            _, pred_sent_reward = (reward_logit[:, 1] >= 0.5).int()
            _, pred_sent_punish = (reward_logit[:, 1] >= 0.5).int()

            output_emotion.extend(pred_emotions)
            output_reward_model_sent.extend(pred_sent_reward)
            output_punish_model_sent.extend(pred_sent_punish)

            ground_truth_emotion.extend(emotion_label)
            ground_truth_sent.extend(sentiment_label)
        
        self.get_accuracy(output_emotion, ground_truth_emotion, output_reward_model_sent, output_punish_model_sent, ground_truth_sent)
        self.get_f1_score(output_emotion, ground_truth_emotion, output_reward_model_sent, output_punish_model_sent, ground_truth_sent)

        
            
            
            
                


        