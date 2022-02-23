import copy

import torch
import h5py

import numpy as np


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


def _get_textual_desc_from_label(se_name, label, classes_names):
    # create textual representation for the ground thruth
    textual_label = {}
    begin, end, begin_name, end_name = -1, -1, "", ""

    from_class = 0
    to_class = 3
    
    if se_name == "HammerThrow":
        from_class = 3
        to_class = 6
    elif se_name == "LongJump":
        from_class = 0
        to_class = 2
    elif se_name == "CleanAndJerk":
        from_class = 6
        to_class = 8
    elif se_name == "ThrowDiscus":
        from_class = 8
        to_class = 10
    
    for c in range(from_class, to_class):
        action_indices = torch.where(label[:, c] == 1)[0].numpy()
    
        if not list(action_indices):
            begin = None
            end = None
        else:
            begin, end = action_indices[0], action_indices[-1]
        
            # offset of 1 (in mnz indices start from 1=
            begin += 1
            end += 1

        begin_name = "b{}".format(classes_names[c])
        end_name = "e{}".format(classes_names[c])
        
        textual_label[begin_name] = begin
        textual_label[end_name] = end
            
    return textual_label
    

def get_textual_label_from_tensor(labels, classes_names_abb):
    labels_keys = labels.keys()
    textual_labels = {}
    for label_key in labels_keys:
        textual_labels[label_key] = {}
        label_key_splitted = label_key.split("-")
        textual_label = _get_textual_desc_from_label(label_key_splitted[1], labels[label_key], classes_names_abb)
    
        textual_labels[label_key] = textual_label
    
    return textual_labels



