import os
import requests
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



def main():
    BASE_PATH = "/home/david/Downloads/landmark-recognition-2021"
    train_df = pd.read_csv(BASE_PATH + "/train.csv")
    # print(train_df.head())
    print(torch.version.cuda)
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()