import copy
import os
import random
import json

import torch
import h5py

import numpy as np
import pandas as pd

# se = structured event


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


def get_avg_actions_durations_in_f(se_name, duration, num_features, avg_actions_durations_s):
    fps = num_features / duration
    avg_actions_durations_f = {}
    
    if se_name == "HighJump":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["HighJump"])
    elif se_name == "LongJump":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["LongJump"])
    elif se_name == "HammerThrow":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["HammerThrow"])

    for action in avg_actions_durations_f.keys():
        avg_action_f = 0
        for i in range(num_features):
            if i / fps >= 0. and i / fps <= avg_actions_durations_f[action]:
                avg_action_f += 1
            else:
                break
        avg_actions_durations_f[action] = avg_action_f
    
    return avg_actions_durations_f


def load_data(mode, path_to_filtered_data, path_to_annotations_json, features):
    # list containing element of the form <video, se_name, duration_s, num_features, interval_se_f, interval_se_s>
    # *_s -> seconds, *_f -> features
    se_list = []

    # load annotations
    with open(path_to_annotations_json, "r") as f:
        annotations = json.load(f)

    # names of se files -> name_of_structured_event.csv
    se_names = os.listdir(path_to_filtered_data)

    for se_name in se_names:
        if "LongJump" not in se_name:
            # get complex events that respect the decomposition in sequence of atomic actions
            filtered_se_df = pd.read_csv(path_to_filtered_data + "{}".format(se_name)).iloc[:, 1:6]
            filtered_se_df = filtered_se_df[filtered_se_df["video"].str.contains(mode)]
            filtered_se_df.columns = ["video", "begin_s", "end_s", "begin_f", "end_f"]

            # get the name of the structured event by removing the extension
            se_name = se_name.split(".")[0]

            # se intervals
            se_intervals = [
                (row[1]["video"], se_name, annotations[row[1]["video"]]["duration"], len(features[row[1]["video"]]),
                 (row[1]["begin_f"], row[1]["end_f"]), (row[1]["begin_s"], row[1]["end_s"])) for row in filtered_se_df.iterrows()]
            
            se_list.extend(se_intervals)

    return se_list


def _get_textual_desc_from_label(se_name, label):
    textual_label = {}
    begin, end, begin_name, end_name = -1, -1, "", ""
    
    if se_name == "HighJump":
        for i in range(3):
            action_indices = torch.where(label[:, i] == 1)[0].numpy()
            
            if not list(action_indices):
                begin = None
                end = None
            else:
                begin, end = action_indices[0], action_indices[-1]
                
                # offset of 1 (in mnz indices start from 1=
                begin += 1
                end += 1
            
            if i == 0:
                begin_name = "bR"
                end_name = "eR"
            elif i == 1:
                begin_name = "bJ"
                end_name = "eJ"
            elif i == 2:
                begin_name = "bF"
                end_name = "eF"
                
            textual_label[begin_name] = begin
            textual_label[end_name] = end
            
    elif se_name == "HammerThrow":
        for i in range(3, 6):
            action_indices = torch.where(label[:, i] == 1)[0].numpy()
            
            if not list(action_indices):
                begin = None
                end = None
            else:
                begin, end = action_indices[0], action_indices[-1]
    
                # offset of 1 (in mnz indices start from 1)
                begin += 1
                end += 1
            
            if i == 3:
                begin_name = "bHT_WU"
                end_name = "eHT_WU"
            elif i == 4:
                begin_name = "bHT_S"
                end_name = "eHT_S"
            elif i == 5:
                begin_name = "bHT_R"
                end_name = "eHT_R"

            textual_label[begin_name] = begin
            textual_label[end_name] = end
            
    return textual_label
    

def get_textual_label_from_tensor(labels):
    labels_keys = labels.keys()
    textual_labels = {}
    for label_key in labels_keys:
        print(label_key)
        textual_labels[label_key] = {}
        label_key_splitted = label_key.split("-")
        textual_label = _get_textual_desc_from_label(label_key_splitted[1], labels[label_key])
    
        textual_labels[label_key] = textual_label
    
    return textual_labels
        
    


