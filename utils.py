import os
import random
import json

import torch
import h5py

import numpy as np
import pandas as pd

# se = structured event

def get_valid_interval(num_features, duration, interval, begin_cut, end_cut):
    # convert from second to features vector
    begin_cut_f, end_cut_f = convert_indices(num_features, duration, begin_cut, end_cut)
    begin_se_f, end_se_f = convert_indices(num_features, duration, interval[0], interval[1])
    
    # maximum clip length is 128, if the cut is > 128 compute a new one
    if end_cut_f - begin_cut_f > 128:
        # length of the structured event
        length_se_f = end_se_f - begin_se_f + 1 # bounds included
        
        # compute a new begin for the cut
        tmp_begin = begin_se_f - 128 + length_se_f
        if begin_se_f - 128 < 0:
            tmp_begin = 0
            
        new_begin_cut_f = random.randint(tmp_begin, begin_se_f)
        partial_length_new_cut = end_se_f - new_begin_cut_f + 1
        # compute the new end for the cut
        new_end_cut_f = new_begin_cut_f + partial_length_new_cut + random.randint(0, 127-partial_length_new_cut)
        
        # covert from features vector to seconds
        fps = num_features / duration
        begin_cut = new_begin_cut_f / fps
        end_cut = new_end_cut_f / fps
        
        # check length of the cut
        assert new_end_cut_f - new_begin_cut_f <= 128
        # check that interval is contained into the new cut
        assert begin_cut <= interval[0] and interval[1] <= end_cut
        print(begin_cut, end_cut)

    return begin_cut, end_cut


def get_data(video, duration, se_name, se_intervals, num_features):
    se_list = []
    
    num_se = len(se_intervals)
    
    # two types of cuts
    begin_cut_1 = 0
    begin_cut_2 = 0
    
    for i in range(0, num_se + 1):
        if (i + 1) % 2 == 0:
            if (i + 1) == (num_se + 1):
                end_cut_1 = duration
            else:
                end_cut_1 = se_intervals[i][0]
            
            begin_cut_1, end_cut_1 = get_valid_interval(num_features, duration, se_intervals[i - 1], begin_cut_1, end_cut_1)
            se_list.append([video, duration, se_name, begin_cut_1, end_cut_1])
            
            if (i + 1) != (num_se + 1):
                begin_cut_1 = se_intervals[i][1]
        else:
            if begin_cut_2 != 0:
                if (i + 1) == (num_se + 1):
                    end_cut_2 = duration
                else:
                    end_cut_2 = se_intervals[i][0]

                begin_cut_2, end_cut_2 = get_valid_interval(num_features, duration, se_intervals[i - 1], begin_cut_2, end_cut_2)
                se_list.append([video, duration, se_name, begin_cut_2, end_cut_2])
            
            if (i + 1) != (num_se + 1):
                begin_cut_2 = se_intervals[i][1]

    return se_list

   
def load_data(path_to_data, path_to_annotations_json, features):
    # list containing element of the form <video, duration, se_name, begin, end>
    se_list = []
    
    # load annotations
    with open(path_to_annotations_json, "r") as f:
        annotations = json.load(f)
    
    # names of se files -> name_of_structured_event.csv
    se_names = os.listdir(path_to_data)
    
    for se_name in se_names:
        df = pd.read_csv(path_to_data + "{}".format(se_name)).iloc[:, 1:4]
        # get the name of the structured event by removing the extension
        se_name = se_name.split(".")[0]
        # take only validation video (to use as training)
        df = df[df["video"].str.contains("validation")]

        df.columns = ["video", "begin", "end"]
        
        # get videos
        videos = list(set(df["video"].to_list()))
        videos.sort()
        
        for v in videos:
            # structured events of the video
            se_of_v = df.groupby("video").get_group(v)
            # duration of the video
            duration_of_v = annotations[v]["duration"]
            # structured events intervals
            se_intervals = [(row[1]["begin"], row[1]["end"]) for row in se_of_v.iterrows()]
    
            se_list.extend(get_data(v, duration_of_v, se_name, se_intervals, len(features[v])))
          
    return se_list


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
    