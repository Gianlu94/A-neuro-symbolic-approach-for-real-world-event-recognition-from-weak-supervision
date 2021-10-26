import argparse

import h5py as h5
import json
import torch

from mlad.configuration import build_config
from mlad.model import build_model

from train import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation of MLAD pretrained model")

    parser.add_argument("--path_to_model", type=str, help="Path to the mlad pre-trained model")
    parser.add_argument("--path_to_conf", type=str, help="Path to configuration files")
    parser.add_argument("--epoch", type=int, help="Epoch of the pretrained model to evaluate")

    args = parser.parse_args()
    
    dataset = "multithumos"
    path_to_model = args.path_to_model
    path_to_conf = args.path_to_conf
    epoch = args.epoch

    # configuration for the dataset
    cfg_dataset = build_config(dataset)

    # load dataset class in a dict {class_name : class_id}
    dataset_classes = {}
    with open(cfg_dataset.classes_file, "r") as cf:
        for row in cf.readlines():
            row_splitted = row.split(" ")
            c_id, c_name = int(row_splitted[0]), row_splitted[1].replace("\n", "")
            dataset_classes[c_name] = c_id
    
    # configuration for the model
    with open(path_to_conf, "r") as jf:
        cfg_model = json.load(jf)
    
    model_version = cfg_model["model_version"]
    eval_mode = cfg_model["eval_mode"]
    num_clips = cfg_model["num_clips"]
    num_mlad_layers = cfg_model["num_mlad_layers"]
    dim_of_features = cfg_model["dim_of_features"]
    f1_threshold = cfg_model["f1_threshold"]
    class_to_evaluate = [dataset_classes[class_name] - 1 for class_name in cfg_model["class_to_evaluate"]]

    use_cuda = True if torch.cuda.is_available() else False

    cfg_model["use_cuda"] = use_cuda

    nn_model = build_model(model_version, num_clips, 65, dim_of_features, num_clips, num_mlad_layers)
    
    # load state of pretrained model
    if use_cuda:
        state = torch.load(path_to_model)
    else:
        state = torch.load(path_to_model, map_location=torch.device('cpu'))

    nn_model.load_state_dict(state["state_dict"])
    
    if use_cuda:
        nn_model.cuda()

    video_list = [line.rstrip().replace('.txt', '') for line in open(cfg_dataset.test_list, 'r').readlines()]
    features_test = h5.File(cfg_dataset.combined_test_file, 'r')
    
    evaluate(cfg_model, cfg_dataset, class_to_evaluate, f1_threshold, epoch, nn_model, features_test, video_list)
    
    
    