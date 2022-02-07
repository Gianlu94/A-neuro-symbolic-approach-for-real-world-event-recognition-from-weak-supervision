import json
import os
import random

import pandas as pd
import torch

from utils import get_avg_actions_durations_in_f


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
                 (row[1]["begin_f"], row[1]["end_f"]), (row[1]["begin_s"], row[1]["end_s"])) for row in
                filtered_se_df.iterrows()]
            
            se_list.extend(se_intervals)
    
    return se_list


data_dec = {}


def _set_nn_value(input):
    if input < 0:
        return 0
    return input


def _set_least_value(v1, v2):
    if v1 >= v2:
        return v2
    else:
        return v1


def get_labels(se_list, cfg_train):
    dec_labels = {}
    
    for example in se_list:
        video, se_name, se_interval = example[0], example[1], example[4]
        
        if se_name not in data_dec:
            data_dec[se_name] = pd.read_csv(cfg_train["path_to_filtered_data"] + se_name + ".csv")
        
        data_dec_se = data_dec[se_name]
        
        intervals_events = []
        
        if se_name == "HighJump":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_hj"] == se_interval[0]) & (
                        data_dec_se["end_f_hj"] == se_interval[1])].copy(deep=True)
            
            begin_ev = _set_nn_value(dec_se["begin_f_run"].values[0] - se_interval[0])
            if begin_ev <= 0:
                dec_se["begin_f_run"] = se_interval[0]
            
            action_duration = dec_se["end_f_run"].values[0] - dec_se["begin_f_run"].values[0]
            
            intervals_events.append([begin_ev, begin_ev + action_duration])
            
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
            
            begin_ev = dec_se["begin_f_fall"].values[0] - se_interval[0]
            intervals_events.append([begin_ev,
                                     begin_ev + _set_least_value(dec_se["end_f_fall"].values[0], se_interval[1]) -
                                     dec_se["begin_f_fall"].values[0]])
            
            labels_indices = list(range(0, 3))
        elif se_name == "HammerThrow":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_ht"] == se_interval[0]) & (
                        data_dec_se["end_f_ht"] == se_interval[1])].copy(deep=True)
            
            begin_ev = _set_nn_value(dec_se["begin_f_ht_wu"].values[0] - se_interval[0])
            
            if begin_ev <= 0:
                dec_se["begin_f_ht_wu"] = se_interval[0]
            
            action_duration = dec_se["end_f_ht_wu"].values[0] - dec_se["begin_f_ht_wu"].values[0]
            
            intervals_events.append([begin_ev, begin_ev + action_duration])
            
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
            
            begin_ev = dec_se["begin_f_ht_r"].values[0] - se_interval[0]
            intervals_events.append(
                [begin_ev, begin_ev + _set_least_value(dec_se["end_f_ht_r"].values[0], se_interval[1]) -
                 dec_se["begin_f_ht_r"].values[0]])
            
            labels_indices = list(range(3, 6))
        
        rows = []
        columns = []
        
        for i in labels_indices:
            begin, end = int(intervals_events[0][0]), int(intervals_events[0][1])
            rows += [i] * (end - begin + 1)
            columns += [j for j in range(begin, end + 1)]
            intervals_events.pop(0)
        
        label_key = "{}-{}-{}".format(video, se_name, se_interval)
        dec_labels[label_key] = torch.zeros((cfg_train["classes"], se_interval[1] - se_interval[0] + 1))
        
        dec_labels[label_key][rows, columns] = 1
        dec_labels[label_key] = dec_labels[label_key].transpose(0, 1)
    
    return dec_labels


# get avg labels for neural baseline
def get_avg_labels(se_list, cfg_train):
    avg_labels = {}
    avg_actions_durations_s = cfg_train["avg_actions_durations_s"]
    classes_names = cfg_train["classes_names"]
    
    for example in se_list:
        video, se_name, duration, num_features, se_interval = \
            example[0], example[1], example[2], example[3], example[4]
        
        # get duration of s
        se_duration = se_interval[1] - se_interval[0] + 1
        
        avg_actions_durations_f = get_avg_actions_durations_in_f(
            se_name, duration, num_features, avg_actions_durations_s)
        
        avg_values = list(avg_actions_durations_f.values())
        tot_avg = sum(avg_values[1:])
        
        label_tensor = torch.zeros((se_duration, len(classes_names)))
        prev_num_frames = 0
        while prev_num_frames != se_duration:
            
            rows, columns = [], []
            inc_action = None
            dec_action = None
            values = avg_values[1:]
            
            # this may happen due to the round operation
            if prev_num_frames == (se_duration - 1):
                # randomly increment one of the actions
                prev_num_frames = 0
                inc_action = random.randint(0, len(values) - 1)
            elif prev_num_frames > se_duration:
                # randomly decrement one of the actions
                prev_num_frames = 0
                dec_action = random.randint(0, len(values) - 1)
            
            for i, avg_value in enumerate(values):
                
                num_frames_to_label = round(se_duration * avg_value / tot_avg)
                
                # increment/decrement action i of one frame (if needed)
                if inc_action is not None and inc_action == i:
                    num_frames_to_label += 1
                if dec_action is not None and dec_action == i:
                    num_frames_to_label -= 1
                
                rows.extend(list(range(prev_num_frames, prev_num_frames + num_frames_to_label)))
                
                if se_name == "HammerThrow":
                    if i == 0:
                        columns.extend([3] * num_frames_to_label)
                    elif i == 1:
                        columns.extend([4] * num_frames_to_label)
                    elif i == 2:
                        columns.extend([5] * num_frames_to_label)
                else:
                    columns.extend([i] * num_frames_to_label)
                
                prev_num_frames += num_frames_to_label
        
        assert len(rows) == len(columns)
        label_tensor[rows, columns] = 1
        avg_labels["{}-{}-{}".format(video, se_name, se_interval)] = label_tensor
    return avg_labels


