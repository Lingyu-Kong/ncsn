import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from easydict import EasyDict
from ncsn.data_loader.data_loader import get_data_loader
from ncsn.models.score_net import CondRefineNetDilated
from ncsn.generator import ScoreBasedGenerator


def main(args_dict):
    train_dataloader, test_dataloader = get_data_loader(dataset=args_dict["dataset"], image_size=args_dict["img_size"], batch_size = args_dict["batch_size"], random_flip=True)
    print(len(train_dataloader.dataset))
    print(len(test_dataloader.dataset))

    scorenet = CondRefineNetDilated(
        logit_transform=args_dict["logit_transform"],
        ngf=args_dict["ngf"],
        num_classes=args_dict["num_classes"],
        num_channels=args_dict["num_channels"],
        image_size=args_dict["img_size"],
    )
    
    if args_dict["optimizer"] == "adam":
       optimizer = optim.Adam(scorenet.parameters(),lr=args_dict["lr"],betas=(args_dict["beta1"], 0.999),weight_decay=args_dict["weight_decay"],amsgrad=args_dict["amsgrad"])
    elif args_dict["optimizer"] == "rmsprop":
       optimizer = optim.RMSprop(scorenet.parameters(),lr=args_dict["lr"],weight_decay=args_dict["weight_decay"])
    elif args_dict["optimizer"] == "sgd":
       optimizer = optim.SGD(scorenet.parameters(),lr=args_dict["lr"],weight_decay=args_dict["weight_decay"])
    else:
       print("Optimizer not supported, use Adam instead") 
       optimizer = optim.Adam(scorenet.parameters(),lr=args_dict["lr"],betas=(args_dict["beta1"], 0.999),weight_decay=args_dict["weight_decay"],amsgrad=args_dict["amsgrad"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorenet.to(device)
    
    generator = ScoreBasedGenerator(
        scorenet=scorenet,
        optimizer=optimizer,
        device=device,
        sm_strategy=args_dict["score_matching_strategy"],
        sigma_begin=args_dict["sigma_begin"],
        sigma_end=args_dict["sigma_end"],
        num_classes=args_dict["num_classes"],
        num_epochs=args_dict["num_epochs"],
        max_step=args_dict["max_step"],
        img_size=args_dict["img_size"],
        num_channels=args_dict["num_channels"],
    )
    
    generator.train(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10") ## Capitalize the dataset name, e.g. MNIST, CIFAR10, SVHN, CELEBA
    parser.add_argument("--score_matching_strategy", type=str, default="dsm")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=500000)
    parser.add_argument("--max_step", type=int, default=200000)
    parser.add_argument("--img_size", type=int, default=32)  ## 28 for MNIST, 32 for CIFAR10
    parser.add_argument("--num_channels", type=int, default=3)  ## 1 for MNIST, 3 for CIFAR10
    parser.add_argument("--logit_transform", action="store_true")
    parser.add_argument("--sigma_begin", type=float, default=1.0)
    parser.add_argument("--sigma_end", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--amsgrad", action="store_true")
    args = parser.parse_args()
    
    args_dict = vars(args)
    main(args_dict)
    