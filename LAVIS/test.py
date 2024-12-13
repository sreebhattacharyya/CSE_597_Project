import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import json
import numpy as np


#Get test dataset
######################################################################
with open('/scratch/bbmr/sbhattacharyya1/projects-src/llm-eval/EmoVIT/EmoVIT/emo/test.json', 'r') as json_file:
    json_data = json.load(json_file)
    
amusement_data = []
anger_data = []
awe_data = []
contentment_data = []
disgust_data = []
excitement_data = []
fear_data = []
sadness_data = []

all_data = []
    
for item in json_data:
    category = item[0]
    if category == 'amusement':
        amusement_data.append(item[1])
    elif category == 'anger':
        anger_data.append(item[1])
    elif category == 'awe':
        awe_data.append(item[1])
    elif category == 'contentment':
        contentment_data.append(item[1])
    elif category == 'disgust':
        disgust_data.append(item[1])
    elif category == 'excitement':
        excitement_data.append(item[1])
    elif category == 'fear':
        fear_data.append(item[1])
    elif category == 'sadness':
        sadness_data.append(item[1])
        
        
all_data = [amusement_data, anger_data, awe_data, contentment_data, disgust_data, excitement_data, fear_data, sadness_data]
emo = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
######################################################################


#Load in model
######################################################################
def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


_, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

load_path = '/scratch/bbmr/sbhattacharyya1/projects-src/llm-eval/EmoVIT/EmoVIT/LAVIS/model_weights1.pth'

model = torch.load(load_path)
model = model.to(device)
model.eval()
######################################################################


#Inference
######################################################################
result = np.zeros((8, 8))
for i in range(8):
    for j in range(len(all_data[i])):
        print(str(i) + '-' + str(j))
            
        raw_image = load_image('/scratch/bbmr/sbhattacharyya1/projects-src/llm-eval/Emo-Set/data/' + all_data[i][j])
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    
        inp = 'Please select the emotion closest to the image from the following options:\
amusement, \
anger, \
awe, \
contentment, \
disgust, \
excitement, \
fear and sadness \
(Do not provide answers outside of the candidates options.) Please answer in the following format:  Predict emotion:'


        text = model.generate({"image": image, "prompt":inp})
        
        answer = text[0].split('\n')[0]
        print(answer)
            
        if 'amusement' in answer.lower():
            result[i][0] = result[i][0] + 1
        elif 'anger' in answer.lower():
            result[i][1] = result[i][1] + 1
        elif 'awe' in answer.lower():
            result[i][2] = result[i][2] + 1
        elif 'contentment' in answer.lower():
            result[i][3] = result[i][3] + 1
        elif 'disgust' in answer.lower():
            result[i][4] = result[i][4] + 1
        elif 'excitement' in answer.lower():
            result[i][5] = result[i][5] + 1
        elif 'fear' in answer.lower():
            result[i][6] = result[i][6] + 1
        elif 'sadness' in answer.lower():
            result[i][7] = result[i][7] + 1
            
print(result)
np.save('/scratch/bbmr/sbhattacharyya1/projects-src/llm-eval/EmoVIT/EmoVIT/LAVIS/result.npy', result)

######################################################################