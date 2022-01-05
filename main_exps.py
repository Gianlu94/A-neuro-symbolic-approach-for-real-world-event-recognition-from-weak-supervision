import argparse
import json
import os
import random

import torch
import h5py as h5

from mlad.configuration import build_config
from mlad.model import build_model
from exp1_mnz_train import train_exp1_mnz
from exp1_neural_train import train_exp1_neural

from utils import load_data, get_avg_actions_durations_in_f
from minizinc.my_functions import build_problem_exp1, fill_mnz_pred_exp1, get_best_sol


# used to get specific structured events from the list
def _filter_data(se_list, se_name):
    filtered_list = []
    for se in se_list:
        if se[1] == se_name:
            filtered_list.append(se)
    
    return filtered_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training ns framework")
    
    parser.add_argument("--path_to_conf", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    dataset = "multithumos"
    
    path_to_conf = args.path_to_conf
    
    use_cuda = True if torch.cuda.is_available() else False

    # build configuration for the dataset
    cfg_dataset = build_config(dataset)
    
    # load configuration (train + model)
    with open(path_to_conf, "r") as jf:
        cfg_train = json.load(jf)
    
    path_to_filtered_data = cfg_train["path_to_filtered_data"]
    model_version = cfg_train["model_version"]
    num_clips = cfg_train["num_clips"]
    num_mlad_layers = cfg_train["num_mlad_layers"]
    dim_of_features = cfg_train["dim_of_features"]

    cfg_train["use_cuda"] = use_cuda
    
    # set seed
    random.seed(cfg_train["seed"])
    torch.manual_seed(cfg_train["seed"])
    
    nn_model = build_model(
        model_version, num_clips, cfg_train["classes"], dim_of_features, num_clips, num_mlad_layers)

    # load pretrained model
    if "path_to_pretrain_model" in cfg_train:
        print("Loading pretrained model \n")
        if use_cuda:
             state = torch.load(cfg_train["path_to_pretrain_model"])
        else:
             state = torch.load(cfg_train["path_to_pretrain_model"], map_location=torch.device('cpu'))
    
        nn_model.load_state_dict(state["state_dict"])
        
    if use_cuda:
        nn_model.cuda()
    
    # train and test features
    features_train = h5.File(cfg_dataset.combined_train_file, 'r')
    features_test = h5.File(cfg_dataset.combined_test_file, 'r')

    # train list of se event
    se_train = load_data("validation", path_to_filtered_data, cfg_dataset.annotations_file, features_train)
    # test list of se event
    se_test = load_data("test", path_to_filtered_data, cfg_dataset.annotations_file, features_test)

    # get a balanced validation set
    se_train_hj = _filter_data(se_train, "HighJump")
    se_val_hj = se_train_hj[160:]
    se_train_hj = se_train_hj[:160]
    se_train_ht = _filter_data(se_train, "HammerThrow")
    se_val_ht = se_train_ht[150:]
    se_train_ht = se_train_ht[:150]
    
    se_train = se_train_hj + se_train_ht
    se_val = se_val_hj + se_val_ht

    if "path_to_mnz_models" in cfg_train:
        path_to_mnz = cfg_train["path_to_mnz_models"]
        # minizinc models for each structured event (se)
        mnz_files_names = os.listdir(path_to_mnz)
        mnz_models = {}
        for mnz_file_name in mnz_files_names:
            se_name = mnz_file_name.split(".")[0]
            if se_name == "HighJump" or se_name == "HammerThrow":
                with open(path_to_mnz + mnz_file_name, "r") as mnz_file:
                    mnz_models[se_name] = mnz_file.read()

        train_exp1_mnz(
            se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset, mnz_models
        )
    else:
        train_exp2_neural(
            se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset
        )
    
    
    
    
