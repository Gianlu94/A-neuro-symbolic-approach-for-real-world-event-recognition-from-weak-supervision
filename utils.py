import os
import json

import torch
import h5py

import numpy as np
import pandas as pd


def load_data(path_to_data, path_to_annotations_json):
    train_list = []

    # load annotations
    with open(path_to_annotations_json, "r") as f:
        annotations = json.load(f)
  
    se_names = os.listdir(path_to_data)
    
    for se_name in se_names:
        df = pd.read_csv(path_to_data + "{}".format(se_name)).iloc[:, 1:4]
        df = df[df["video"].str.contains("validation")]
        df.columns = ["video", "begin", "end"]
        
        for row in df.iterrows():
            row = row[1]
            se_name = se_name.split(".")[0]
            train_list.append([row["video"], annotations[row["video"]]["duration"], se_name, row["begin"], row["end"]])
            
    
    return train_list


def convert_to_float_tensor(input):
    if isinstance(input, list):
        if torch.cuda.is_available():
            input = [torch.cuda.FloatTensor(el) for el in input]
        else:
            input = [torch.FloatTensor(el) for el in input]
    elif isinstance(input, dict):
        for key, el in input.item():
            if torch.cuda.is_available():
                input[key] = torch.cuda.FloatTensor(el)
            else:
                input[key] = torch.FloatTensor(el)
    elif isinstance(input, h5py._hl.files.File):
        input_dict = {}
        for key, el in input.items():
            if torch.cuda.is_available():
                input_dict[key] = torch.cuda.FloatTensor(np.array(el))
            else:
                input_dict[key] = torch.FloatTensor(np.array(el))
        input = input_dict
    else:
        if torch.cuda.is_available():
            input = torch.cuda.FloatTensor(input)
        else:
            input = torch.FloatTensor(input)
    
    return input

def convert_indices(num_features, duration, begin_s, end_s):
    # convert begin and end from seconds to features vectors indices
    fps = num_features / duration
    begin_f = round(begin_s * fps)
    end_f = round(end_s * fps)
    
    return begin_f, end_f
    