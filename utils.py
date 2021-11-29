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
    
    if se_name == "high_jump":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["HighJump"])
    elif se_name == "long_jump":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["LongJump"])
    
    for action in avg_actions_durations_f.keys():
        avg_action_f = 0
        for i in range(num_features):
            if i / fps >= 0. and i / fps <= avg_actions_durations_f[action]:
                avg_action_f += 1
            else:
                break
        avg_actions_durations_f[action] = avg_action_f
    
    return avg_actions_durations_f


def insert_intervals_in_features(se_of_v, begin_s, end_s, fps, num_features):
    begin_f = []
    end_f = []
    for i in range(len(begin_s)):
        begin_a1 = begin_s[i]
        end_a1 = end_s[i]
        range_features = []
        for j in range(num_features):
            if j/fps >= begin_a1 and j/fps <= end_a1:
                range_features.append(j)
        begin_f.append(range_features[0])
        end_f.append(range_features[-1])

    se_of_v.insert(len(se_of_v.columns), "begin_f", begin_f)
    se_of_v.insert(len(se_of_v.columns), "end_f", end_f)


def get_data(video, duration_of_v, num_features, se_name, se_intervals):
    se_list = []
    
    num_se = len(se_intervals)
    
    # two types of cuts
    begin_cut_1 = 0
    begin_cut_2 = 0
    
    for i in range(0, num_se + 1):
        if (i + 1) % 2 == 0:
            if (i + 1) == (num_se + 1):
                end_cut_1 = num_features
            else:
                end_cut_1 = se_intervals[i][0] - 1
            
            se_list.append([
                video, duration_of_v, num_features, se_name,
                (begin_cut_1, end_cut_1), se_intervals[i-1][:2], se_intervals[i-1][2:]])
            
            if (i + 1) != (num_se + 1):
                begin_cut_1 = se_intervals[i][1] + 1
        else:
            if begin_cut_2 != 0:
                if (i + 1) == (num_se + 1):
                    end_cut_2 = num_features
                else:
                    end_cut_2 = se_intervals[i][0] - 1

                se_list.append([
                    video, duration_of_v, num_features, se_name,
                    (begin_cut_2, end_cut_2), se_intervals[i-1][:2], se_intervals[i-1][2:]])
            
            if (i + 1) != (num_se + 1):
                begin_cut_2 = se_intervals[i][1] + 1

    return se_list


def filter_se_list(se_list, filtered_se_df):
    filtered_list = []
    for sample in se_list:
        columns = filtered_se_df.columns
        video = sample[0]
        # se in seconds
        begin_s, end_s = sample[-1][0], sample[-1][1]
        # if this a se that decomposes in sequence of atomic atomic action keep it
        if ((filtered_se_df[columns[0]] == video) & (filtered_se_df[columns[1]] == begin_s) & (filtered_se_df[columns[2]] == end_s)).any():
            filtered_list.append(sample)
            
    return filtered_list


def compute_valid_cut(se_list):
    for i, se in enumerate(se_list):
        cut = se[4]
        event = se[5]
        tmp_event = list(event)

        if (cut[1] - cut[0]) + 1 > 128:
            new_cut = (random.randint(cut[0], tmp_event[0]), random.randint(tmp_event[1], cut[1]))
            while (new_cut[1] - new_cut[0]) + 1 > 128:
                new_cut = [random.randint(cut[0], tmp_event[0]), random.randint(tmp_event[1], cut[1])]
            se_list[i][4] = tuple(new_cut)
            
 
def load_data(mode, path_to_data, path_to_filtered_data, path_to_annotations_json, features):
    # list containing element of the form <video, duration_s, num_features, se_name, cut_f, event_f, event_s>
    # *_s -> seconds, *_f -> features
    se_list = []
    
    # load annotations
    with open(path_to_annotations_json, "r") as f:
        annotations = json.load(f)
    
    # names of se files -> name_of_structured_event.csv
    se_names = os.listdir(path_to_data)
    
    for se_name in se_names:
        # get all complex events
        all_se_df = pd.read_csv(path_to_data + "{}".format(se_name)).iloc[:, 1:4]
        # get complex events that respect the decomposition in sequence of atomic actions
        filtered_se_df = pd.read_csv(path_to_filtered_data + "{}".format(se_name)).iloc[:, 1:4]

        # get the name of the structured event by removing the extension
        se_name = se_name.split(".")[0]
        # take validation or testing video (to use as training)
        all_se_df = all_se_df[all_se_df["video"].str.contains(mode)]

        all_se_df.columns = ["video", "begin", "end"]
        
        # get videos
        videos = list(set(all_se_df["video"].to_list()))
        videos.sort()
        
        for v in videos:
            # duration of the video in s
            duration_of_v = annotations[v]["duration"]
            # num_features
            num_features = len(features[v])

            # se of the video
            se_of_v = all_se_df.groupby("video").get_group(v)
            fps = len(features[v]) / duration_of_v
            
            # insert begin and end of actions expressed in features
            insert_intervals_in_features(se_of_v, se_of_v["begin"].to_list(), se_of_v["end"].to_list(), fps, num_features)
            #se_of_v.insert(len(se_of_v.columns), "begin_f", (se_of_v["begin"] * fps).astype(int).to_list())
            #se_of_v.insert(len(se_of_v.columns), "end_f", (se_of_v["end"] * fps).astype(int).to_list())
    
            # order actions
            se_of_v = se_of_v.sort_values(by=["begin_f"])
            # se intervals
            se_intervals = [(row[1]["begin_f"], row[1]["end_f"], row[1]["begin"], row[1]["end"]) for row in se_of_v.iterrows()]
            
            se_list_tmp = get_data(v, duration_of_v, num_features, se_name, se_intervals)

            se_list.extend(filter_se_list(se_list_tmp, filtered_se_df))
    
    # resize cuts bigger than 128
    compute_valid_cut(se_list)

    return se_list

