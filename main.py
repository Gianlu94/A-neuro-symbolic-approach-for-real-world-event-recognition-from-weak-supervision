import argparse
import json
import os
import random

import torch
import h5py as h5

from mlad.configuration import build_config
from mlad.model import build_model
from train import train_model
from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training ns framework")
    
    # terminals' arguments
    parser.add_argument("--path_to_model", type=str, help="Path to the mlad pre-trained model")
    parser.add_argument("--path_to_conf", type=str, help="Path to configuration file")
    parser.add_argument("--path_to_mzn", type=str, help="Path to minizinc models")
    parser.add_argument("--path_to_data", type=str, help="Path to train data")
    
    args = parser.parse_args()
    
    dataset = "multithumos"
    path_to_model = args.path_to_model
    path_to_conf = args.path_to_conf
    path_to_mzn = args.path_to_mzn
    path_to_data = args.path_to_data

    use_cuda = True if torch.cuda.is_available() else False

    # build configuration for the dataset
    cfg_dataset = build_config(dataset)
    
    # load dataset classes in a dict {class_name : class_id}
    dataset_classes = {}
    with open(cfg_dataset.classes_file, "r") as cf:
        for row in cf.readlines():
            row_splitted = row.split(" ")
            c_id, c_name = int(row_splitted[0]), row_splitted[1].replace("\n", "")
            dataset_classes[c_name] = c_id
        
    
    # load configuration for the model
    with open(path_to_conf, "r") as jf:
        cfg_model = json.load(jf)
    
    model_version = cfg_model["model_version"]
    eval_mode = cfg_model["eval_mode"]
    num_clips = cfg_model["num_clips"]
    num_mlad_layers = cfg_model["num_mlad_layers"]
    dim_of_features = cfg_model["dim_of_features"]

    cfg_model["use_cuda"] = use_cuda
    
    # set seed
    random.seed(cfg_model["seed"])
    torch.manual_seed(cfg_model["seed"])
    
    nn_model = build_model(model_version, num_clips, 65, dim_of_features, num_clips, num_mlad_layers)

    # load pretrained model
    if use_cuda:
        state = torch.load(path_to_model)
    else:
        state = torch.load(path_to_model, map_location=torch.device('cpu'))

    nn_model.load_state_dict(state["state_dict"])
    
    # minizinc models for each structured event (se)
    mnz_files_names = os.listdir(path_to_mzn)
    mnz_models = {}
    for mnz_file_name in mnz_files_names:
        se_name = mnz_file_name.split(".")[0]
        with open(path_to_mzn + mnz_file_name, "r") as mzn_file:
            mnz_models[se_name] = mzn_file.read()

    if use_cuda:
        nn_model.cuda()
    
    # train and test features
    features_train = h5.File(cfg_dataset.combined_train_file, 'r')
    features_test = h5.File(cfg_dataset.combined_test_file, 'r')

    # train list of se event
    # se_train = [["video_validation_0000361", 696.47, "high_jump", 101.9, 106.8],...,
    #             ["video_validation_0000361", 696.47, "high_jump", 112.5, 116.5]]
    se_train = load_data(path_to_data, cfg_dataset.annotations_file, features_train)

    train_model(cfg_dataset, cfg_model, dataset_classes, se_train, features_train, features_test, nn_model, mnz_models)
    
    
    
