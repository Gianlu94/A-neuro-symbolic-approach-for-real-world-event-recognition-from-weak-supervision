import json
import os
import random

import pandas as pd
import torch


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
        # get complex events that respect the decomposition in sequence of atomic actions
        filtered_se_df = pd.read_csv(path_to_filtered_data + "{}".format(se_name), encoding='utf-7').iloc[:, 1:6]
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


# used to get specific structured events from the list
def filter_data(se_list, se_name):
    filtered_list = []
    for se in se_list:
        if se[1] == se_name:
            filtered_list.append(se)

    return filtered_list


def get_validation_set(all_se_train, se_names, ratio, seed=0):
    train_split = []
    val_split = []
    
    rng = random.Random(seed)
    for se_name in se_names:
        se_list = filter_data(all_se_train, se_name)
        num_train_ex = len(se_list)
        num_val_ex = round(num_train_ex * ratio)
        
        val_exs = rng.sample(se_list, num_val_ex)
        train_split += list(set(se_list) - set(val_exs))
        val_split += val_exs
        
    train_split.sort()
    return train_split, val_split

   
data_dec = {}


def get_labels(se_list, cfg_train):
    dec_labels = {}
    classes = cfg_train["classes"]
    if isinstance(classes, list):
        classes = classes[1]
        
    for example in se_list:
        video, se_name, se_interval = example[0], example[1], example[4]
        
        if se_name not in data_dec:
            data_dec[se_name] = pd.read_csv(cfg_train["path_to_filtered_data"] + se_name + ".csv", encoding='utf-7')

        data_dec_se = data_dec[se_name]
        
        intervals_events = []
        
        if se_name == "HighJump":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_hj"] == se_interval[0]) & (
                        data_dec_se["end_f_hj"] == se_interval[1])].copy(deep=True)
            
            begin_ev = dec_se["begin_f_run"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_run"].values[0] - dec_se["begin_f_run"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])
            
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
            
            begin_ev = dec_se["begin_f_fall"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f_fall"].values[0] - dec_se["begin_f_fall"].values[0]])
            
            labels_indices = [0, 1, 2]
        elif se_name == "LongJump":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_lj"] == se_interval[0]) & (
                        data_dec_se["end_f_lj"] == se_interval[1])].copy(deep=True)

            begin_ev = dec_se["begin_f_run"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_run"].values[0] - dec_se["begin_f_run"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])

            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])

            begin_ev = dec_se["begin_f_sit"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f_sit"].values[0] - dec_se["begin_f_sit"].values[0]])

            labels_indices = [0, 1, 3]
        elif se_name == "PoleVault":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_pv"] == se_interval[0]) & (
                        data_dec_se["end_f_pv"] == se_interval[1])].copy(deep=True)

            begin_ev = dec_se["begin_f_run"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_run"].values[0] - dec_se["begin_f_run"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])

            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])

            begin_ev = dec_se["begin_f_jump"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f_jump"].values[0] - dec_se["begin_f_jump"].values[0]])

            begin_ev = dec_se["begin_f_fall"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f_fall"].values[0] - dec_se["begin_f_fall"].values[0]])

            labels_indices = [0, 4, 1, 2]
            
        elif se_name == "HammerThrow":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_ht"] == se_interval[0]) & (
                        data_dec_se["end_f_ht"] == se_interval[1])].copy(deep=True)
            
            begin_ev = dec_se["begin_f_ht_wu"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_ht_wu"].values[0] - dec_se["begin_f_ht_wu"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])
            
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
            
            begin_ev = dec_se["begin_f_ht_r"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f_ht_r"].values[0] -
                 dec_se["begin_f_ht_r"].values[0]])
            
            labels_indices = [5, 6, 7]
        elif se_name == "ThrowDiscus":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_td"] == se_interval[0]) & (
                        data_dec_se["end_f_td"] == se_interval[1])].copy(deep=True)
    
            begin_ev = dec_se["begin_f_td_wu"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_td_wu"].values[0] - dec_se["begin_f_td_wu"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])
    
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
    
            labels_indices = [8, 9]
        elif se_name == "Shotput":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_sp"] == se_interval[0]) & (
                        data_dec_se["end_f_sp"] == se_interval[1])].copy(deep=True)
    
            begin_ev = dec_se["begin_f_spb"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_spb"].values[0] - dec_se["begin_f_spb"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])
    
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
    
            labels_indices = [10, 11]
        elif se_name == "JavelinThrow":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_jt"] == se_interval[0]) & (
                        data_dec_se["end_f_jt"] == se_interval[1])].copy(deep=True)
    
            begin_ev = dec_se["begin_f_run"].values[0] - se_interval[0]
            action_duration = dec_se["end_f_run"].values[0] - dec_se["begin_f_run"].values[0]
            intervals_events.append([begin_ev, begin_ev + action_duration])
    
            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])
    
            labels_indices = [0, 11]
        
        rows = []
        columns = []

        for i in labels_indices:
            begin, end = int(intervals_events[0][0]), int(intervals_events[0][1])
            rows += [i] * (end - begin + 1)
            columns += [j for j in range(begin, end + 1)]
            intervals_events.pop(0)
        
        label_key = "{}-{}-{}".format(video, se_name, se_interval)
    
        dec_labels[label_key] = torch.zeros((classes, se_interval[1] - se_interval[0] + 1))
        
        dec_labels[label_key][rows, columns] = 1
        dec_labels[label_key] = dec_labels[label_key].transpose(0, 1)
    
    return dec_labels


# get avg labels for neural baseline
def get_avg_labels(se_list, cfg_train):
    avg_labels = {}
    seed = cfg_train["seed"]
    avg_actions_durations_f = cfg_train["avg_actions_durations_f"]
    classes_names = cfg_train["classes_names"]
    structured_events = cfg_train["structured_events"]
    is_nn_for_ev = cfg_train["is_nn_for_ev"]

    rng = random.Random(seed)
    for example in se_list:
        video, se_name, duration, num_features, se_interval = \
            example[0], example[1], example[2], example[3], example[4]
        
        # get duration of s
        se_duration = (se_interval[1] - se_interval[0]) + 1

        for current_se in structured_events:
            if current_se != "StructuredJump" and current_se != "StructuredThrow":
                avg_values = list(avg_actions_durations_f[current_se].values())
                tot_avg = sum(avg_values)
                
                label_tensor = torch.zeros((se_duration, len(classes_names)))
                prev_num_frames = 0
        
                while prev_num_frames != se_duration:
                    rows, columns = [], []
                    inc_action = None
                    dec_action = None
                    values = avg_values
                    
                    # this may happen due to the round operation
                    if prev_num_frames == (se_duration - 1):
                        # randomly increment one of the actions
                        prev_num_frames = 0
                        inc_action = rng.randint(0, len(values) - 1)
                    # elif prev_num_frames > se_duration:
                    #     breakpoint()
                    #     # randomly decrement one of the actions
                    #     prev_num_frames = 0
                    #     dec_action = rng.randint(0, len(values) - 1)
                    
                    for i, avg_value in enumerate(values):
                        
                        num_frames_to_label = round(se_duration * avg_value / tot_avg)
                        
                        # increment/decrement action i of one frame (if needed)
                        if inc_action is not None and inc_action == i:
                            num_frames_to_label += 1
                        if dec_action is not None and dec_action == i:
                            num_frames_to_label -= 1
                        
                        rows.extend(list(range(prev_num_frames, prev_num_frames + num_frames_to_label)))
                        
                        if current_se == "LongJump":
                            if i == 2:
                                columns.extend([3] * num_frames_to_label)
                            else:
                                columns.extend([i] * num_frames_to_label)
                        elif current_se == "PoleVault":
                            if i == 1:
                                columns.extend([4] * num_frames_to_label)
                            elif i == 2:
                                columns.extend([1] * num_frames_to_label)
                            elif i == 3:
                                columns.extend([2] * num_frames_to_label)
                            else:
                                columns.extend([i] * num_frames_to_label)
                        elif current_se == "HammerThrow":
                            if i == 0:
                                columns.extend([5] * num_frames_to_label)
                            elif i == 1:
                                columns.extend([6] * num_frames_to_label)
                            elif i == 2:
                                columns.extend([7] * num_frames_to_label)
                        elif current_se == "ThrowDiscus":
                            if i == 0:
                                columns.extend([8] * num_frames_to_label)
                            elif i == 1:
                                columns.extend([9] * num_frames_to_label)
                        elif current_se == "Shotput":
                            if i == 0:
                                columns.extend([10] * num_frames_to_label)
                            elif i == 1:
                                columns.extend([11] * num_frames_to_label)
                        elif current_se == "JavelinThrow":
                            if i == 1:
                                columns.extend([11] * num_frames_to_label)
                            else:
                                columns.extend([i] * num_frames_to_label)
                        else:
                            columns.extend([i] * num_frames_to_label)
                        
                        prev_num_frames += num_frames_to_label
        
                    if prev_num_frames > se_duration:
                        prev_num_frames = se_duration
                        rows = rows[:se_duration]
                        columns = columns[:se_duration]
                    
                assert len(rows) == len(columns)
                label_tensor[rows, columns] = 1
                # set to 1 structured events
                if is_nn_for_ev == 2:
                    label_tensor[:, 12+structured_events[current_se]] = 1
                    
                clip_key = "{}-{}-{}".format(video, se_name, se_interval)
                if clip_key not in avg_labels:
                    avg_labels[clip_key] = {}
                avg_labels[clip_key][current_se] = label_tensor
    return avg_labels


def get_se_labels(se_list, cfg_train):
    structured_events = cfg_train["structured_events"]
    num_se = cfg_train["classes"]
    if isinstance(num_se, list):
        num_se = num_se[0]
    labels = {}
    
    for example in se_list:
        video, se_name, _, _, se_interval = example[0], example[1], example[2], example[3], example[4]
        clip_key = "{}-{}-{}".format(video, se_name, se_interval)

        se_duration = (se_interval[1] - se_interval[0]) + 1
        
        for current_se, idx in structured_events.items():
            
            label_tensor = torch.zeros(se_duration, num_se)
            label_tensor[:, idx] = 1.
            
            if clip_key not in labels:
                labels[clip_key] = {}
            
            labels[clip_key][current_se] = label_tensor

    return labels


def get_examples_direct_supervision(se_list, se_dir_sup, seed=0):
    ds_perc = [i / 100 for i in range(10, 110, 10)]
    # get examples for which we are going to use direct supervision
    
    examples_dir_sup = []
    # if not empty
    if se_dir_sup:
        se_to_filter = list(se_dir_sup.keys())
        examples_per_se = {}
        
        for example in se_list:
            example_class = example[1]
            if example_class in se_to_filter:
                examples_per_se.setdefault(example_class, []).append(example)
        
        rng = random.Random(seed)

        num_examples_for_se = {se: len(examples) for se, examples in examples_per_se.items()}
        prev_num_of_examples = {se: 0 for se, _ in examples_per_se.items()}

        # for each perc
        for current_perc_of_examples in ds_perc:
            for se, examples in examples_per_se.items():
                num_examples = num_examples_for_se[se]

                # for each percentage calculate the additional examples to take with respect to the previous percentage
                num_examples_to_take = round(num_examples * current_perc_of_examples) - prev_num_of_examples[se]
                print("Total {} - Take {} out of {} -- perc{:.2}".format(num_examples, num_examples_to_take, len(examples), current_perc_of_examples))
                examples_to_take = rng.sample(examples, num_examples_to_take)
                examples_dir_sup.extend(examples_to_take)

                for example_to_remove in examples_to_take:
                    examples.remove(example_to_remove)
                
                prev_num_of_examples[se] += num_examples_to_take

            # get the percentage of direct supervision to use
            perc_of_examples = se_dir_sup[se]
            
            # reached the perc of examples
            if current_perc_of_examples == perc_of_examples:
                break
    
    return examples_dir_sup
    
    


