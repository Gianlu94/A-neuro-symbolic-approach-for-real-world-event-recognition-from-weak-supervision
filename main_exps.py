import argparse
import json
import os
import random

import torch
import h5py as h5

from mlad.configuration import build_config
from mlad.model import build_model
from exp1_mnz import train_exp1_mnz
from exp2_mnz import train_exp2_mnz
from exp1_baselines import train_exp1_neural, evaluate_test_set_with_proportion_rule_on_aa

from dataset import load_data, get_validation_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training ns framework")
    
    parser.add_argument("-path_to_conf", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    dataset = "multithumos"
    
    path_to_conf = args.path_to_conf
    
    use_cuda = True if torch.cuda.is_available() else False

    # build configuration for the dataset
    cfg_dataset = build_config(dataset)
    
    # load configuration (train + model)
    with open(path_to_conf, "r") as jf:
        cfg_train = json.load(jf)
    
    exp_num, exp_type = cfg_train["exp"].split("-")
    exp_num = int(exp_num)
    path_to_filtered_data = cfg_train["path_to_filtered_data"]
    model_version = cfg_train["model_version"]
    num_clips = cfg_train["num_clips"]
    num_mlad_layers = cfg_train["num_mlad_layers"]
    dim_of_features = cfg_train["dim_of_features"]

    cfg_train["use_cuda"] = use_cuda
    seed = cfg_train["seed"]
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    
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

    se_train, se_val = get_validation_set(
        se_train, list(cfg_train["structured_events"].keys()), cfg_train["val_ratio"], seed
    )

    if exp_type == "mnz":
        path_to_mnz = cfg_train["path_to_mnz_models"]
        # minizinc models for each structured event (se)
        mnz_files_names = os.listdir(path_to_mnz)
        mnz_models = {}
        for mnz_file_name in mnz_files_names:
            se_name = mnz_file_name.split(".")[0]
            with open(path_to_mnz + mnz_file_name, "r") as mnz_file:
                mnz_models[se_name] = mnz_file.read()

        if exp_num == 1:
            train_exp1_mnz(
                se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset, mnz_models
            )
        elif exp_num == 2:
            train_exp2_mnz(
                se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset, mnz_models
            )
    elif exp_type == "neural_baseline":
        train_exp1_neural(
            se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset
        )
    elif exp_type == "proportion_rule_baseline":
        evaluate_test_set_with_proportion_rule_on_aa(se_test, cfg_train, cfg_dataset)
    else:
        print("ERROR: Experiment {} not found".format(exp_type))
    
    
    
    
