import os
import pandas 
import torch 
import argparse

from torch.utils.data import random_split, DataLoader

from src.options import DefaultArgs
from src.data import DataSetup
from src.models import RewardPunishModel
from src.trainer import Trainer

def main(): 
    parser = argparse.ArgumentParser(description='Choice of Dataset, mode (training/evaluation).')
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of the dataset to train or test on. Options: emoset')
    parser.add_argument('-m', '--mode', type=str, required=True, help='Mode of running: can be train or test.')

    print("Parsing arguments...")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    mode = args.mode

    print(f"Dataset name received: {dataset_name}")
    print(f"Mode received = {mode}")
    print()

    defaults_args = DefaultArgs(exp_id = 'Default') # change exp_id

    # distributed training setup 

    # logger setup 

    print("Creating dataset...")

    dataset = DataSetup(dataset_name, defaults_args)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    test_size = int(0.2 * dataset_size)
    val_size = dataset_size - (train_size + test_size)
    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    print("Dataset and dataloaders created...")
    print()

    trainloader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=defaults_args.train_load_workers
    )
    valloader = DataLoader(
        val,
        batch_size=32,
        shuffle=False,
        num_workers=defaults_args.val_load_workers
    )
    testloader = DataLoader(
        test,
        batch_size=32,
        shuffle=False,
        num_workers=defaults_args.test_load_workers
    )

    print("Creating model...")
    model = RewardPunishModel(defaults_args.device, "blip", 8)
    print("Model created...")
    print()

    print("Creating Trainer...")
    trainer = Trainer(trainloader, 
                      valloader,
                      testloader,
                      defaults_args, 
                      model,
                      defaults_args.device)
    
    if mode == 'train':
        print("Beginning training...")
        print()
        trainer.train()
    if mode == 'test': 
        print("Beginning testing...")
        print()
        trainer.test()
    
    
    
if __name__ == '__main__': 
    main()

