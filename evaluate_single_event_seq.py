import argparse
import copy
import json
import os
import pickle
import random
import time

import h5py as h5
import numpy as np
import pymzn
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
import torch
from torch.autograd.variable import Variable
from torch import nn
from tqdm import tqdm

from minizinc.my_functions import build_problem
from mlad.configuration import build_config
from mlad.model import build_model
from train import train_model, build_labels
from utils import load_data, convert_indices


avg_actions_durations_s = {
    "HighJump": {
        "avgHJ": 4.2624,
        "avgR": 2.3675,
        "avgJ": 0.6512,
        "avgF": 0.8319,
    },
    "LongJump": {
        "avgLJ": 6.9693,
        "avgR": 3.6600,
        "avgJ": 1.3041,
        "avgS": 2.3105,
    }
}


def get_flatted_list(list):
    flatted_list = [item for sublist in list for item in sublist]
    return flatted_list


def fill_mnz_pred(mnz_pred, sol, se_name, dataset_classes):
    if se_name == "HighJump":
        class_of_interest = [
            dataset_classes["HighJump"] - 1,
            dataset_classes["Run"] - 1, dataset_classes["Jump"] - 1, dataset_classes["Fall"] - 1
        ]
    elif se_name == "LongJump":
        class_of_interest = [
            dataset_classes["LongJump"] - 1,
            dataset_classes["Run"] - 1, dataset_classes["Jump"] - 1, dataset_classes["Sit"] - 1
        ]
    
    time_points = list(sol[0].values())
    # index start from 0
    time_points = [t - 1 for t in time_points]
    
    rows = [list(range(time_points[i], time_points[i + 1] + 1)) for i in range(0, len(time_points), 2)]
    columns = [len(r) * [class_of_interest[i]] for i, r in enumerate(rows)]
    rows = get_flatted_list(rows)
    columns = get_flatted_list(columns)
    
    assert len(rows) == len(columns)
    mnz_pred[rows, columns] = 1


def save_filtered_outputs(sample, eval_type, filtered_outputs, filtered_labels, filtered_outputs_to_save):
    if sample not in (filtered_outputs_to_save):
        filtered_outputs_to_save[sample] = {}
    if "ground_truth" not in filtered_outputs_to_save[sample]:
        filtered_outputs_to_save[sample]["ground_truth"] = filtered_labels.transpose(0, 1).data.numpy()
    if eval_type not in filtered_outputs_to_save[sample]:
        filtered_outputs_to_save[sample][eval_type] = filtered_outputs.transpose(0, 1).data.numpy()


def get_avg_actions_durations_in_f(se_name, duration, features):
    fps = features / duration
    avg_actions_durations_f = {}

    if se_name == "high_jump":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["HighJump"])
    elif se_name == "long_jump":
        avg_actions_durations_f = copy.deepcopy(avg_actions_durations_s["LongJump"])

    for action in avg_actions_durations_f.keys():
        avg_actions_durations_f[action] = int(avg_actions_durations_f[action] * fps)
            
    return avg_actions_durations_f

   
def get_best_sol(sols, criteria, output, dataset_classes):
    scores = torch.zeros(len(sols))
    actions = []
    for idx_sol, sol in enumerate(sols):
        begin_name = list(sol[0].keys())[0]
        if "HJ" in begin_name:
            action = "HighJump"
        elif "LJ" in begin_name:
            action = "LongJump"
        actions.append(action)
        
        mnz_pred = list(sol[0].values())
        se_begin, se_end = mnz_pred[0], mnz_pred[1]
        if criteria == "max_avg":
            scores[idx_sol] = torch.mean(output[dataset_classes[action] - 1, se_begin:se_end+1])
    
    idx_max_action = torch.argmax(scores)
    
    return sols[idx_max_action], actions[idx_max_action]
        
    
def evaluate(eval_type, cfg_model, cfg_dataset, epoch, nn_model, features_test, se_test, mnz_models, criteria):
    type_models = ""
    if "avg" in eval_type:
        eval_type = "nmnz"
        type_models = "avg"
        
    # set evaluation mode
    nn_model.eval()
    
    num_clips = cfg_model["num_clips"]
    use_cuda = cfg_model["use_cuda"]
    f1_threshold = cfg_model["f1_threshold"]
    class_to_evaluate = cfg_model["class_to_evaluate"]
    path_to_nn_output = args.path_to_nn_output

    # later used to get plots
    filtered_outputs_to_save = {}
    path_to_visual_results = "./visual_results/"
    os.makedirs(path_to_visual_results, exist_ok=True)

    path_to_filtered_outputs = path_to_visual_results + "filtered_outputs_{}.pickle".format(cfg_model["mnz_models"])
    if os.path.exists(path_to_filtered_outputs):
        with open(path_to_filtered_outputs, "rb") as ffo:
            filtered_outputs_to_save = pickle.load(ffo)

    num_mnz_models = len(mnz_models.keys())
    
    # check if nn output has been already computed
    nn_output = {}
    os.makedirs(path_to_nn_output, exist_ok=True)
    path_to_nn_output_file = path_to_nn_output + "outputs_{}.pickle".format(num_clips)
    if os.path.exists(path_to_nn_output_file):
        with open(path_to_nn_output_file, "rb") as pf:
            nn_output = pickle.load(pf)

    predictions, ground_truth = [], []

    tot_time = 0.
    # iterate on the events
    for i, sample_test in enumerate(se_test):
        breakpoint()
        duration = sample_test[1]
        num_features = sample_test[2]
        
        # se happening in the clip
        cut = sample_test[4]
        event = sample_test[5]
        begin_se_c = event[0] - cut[0]
        end_se_c = begin_se_c + event[1] - event[0]
        
        sample_test.append((begin_se_c, end_se_c))
        
        print("\nProcessing sample [{}, {}, {}]  {}/{}  ".format(sample_test[0], sample_test[3], sample_test[-1],
                                                                  i + 1, len(se_test)), end="")
        if str(sample_test) not in nn_output:
            
            video, se_name, interval_cut_f = sample_test[0], sample_test[3], sample_test[4]

            # get features for the current video
            features_video = features_test[video]
            features_video = np.array(features_video)
            features_video = Variable(torch.from_numpy(features_video).type(torch.FloatTensor))

            # get labels
            labels_video = build_labels(
                video, cfg_dataset.annotations_file, len(features_video), cfg_dataset.num_classes, False)
    
            labels_video = Variable(torch.from_numpy(labels_video).type(torch.FloatTensor))
            
            # get clip and its labels
            features_clip = features_video[interval_cut_f[0]:interval_cut_f[1]+1]
            labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1]+1]
            with torch.no_grad():
                if num_clips > 0:
                    if len(features_clip) < num_clips:
                        # padding
                        features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
                        labels_to_append = torch.zeros(num_clips - len(labels_clip) % num_clips, labels_clip.shape[1])
                        features_clip = torch.cat((features_clip, features_to_append), 0)
                        labels_clip = torch.cat((labels_clip, labels_to_append), 0)
                    assert len(features_clip) > 0
                else:
                    features_clip = torch.unsqueeze(features_clip, 0)
                    labels_clip = torch.unsqueeze(labels_clip, 0)
            
                if use_cuda:
                    features_clip = features_clip.cuda()
                    labels_clip = labels_clip.cuda()

                # get the output from the network
                out = nn_model(features_clip)
                outputs = out['final_output']

                outputs = nn.Sigmoid()(outputs)
                # save network outputs
                nn_output[str(sample_test)] = (outputs, labels_clip)
        else:
            outputs = nn_output[str(sample_test)][0]
            labels_clip = nn_output[str(sample_test)][1]

        outputs = outputs.reshape(-1, 65)
        
        if eval_type == "nmnz":
            tot_time_sample = 0.
            mnz_pred = torch.zeros(labels_clip.shape)
            output_transpose = outputs.transpose(0, 1)
            sols = []
            for se_name, mnz_model in mnz_models.items():
                avg_actions_durations_f = None
                if type_models == "avg":
                    avg_actions_durations_f = get_avg_actions_durations_in_f(se_name, duration, num_features)
                mnz_problem, _ = build_problem(se_name, mnz_model, output_transpose, dataset_classes, avg_actions_durations_f)
                start_time = time.time()
                sol = pymzn.minizinc(mnz_problem, solver=pymzn.Chuffed())
                end_time = time.time()

                tot_time_sample += end_time - start_time
                sols.append(sol)
            
            # get best solution
            best_sol, se_name = get_best_sol(sols, criteria, output_transpose, dataset_classes)
            fill_mnz_pred(mnz_pred, best_sol, se_name, dataset_classes)
            
            print("--- ({} calls to mnz) -- tot_time = {:.2f} - avg_time = {:.2f} \n".format(
                num_mnz_models, tot_time_sample, tot_time_sample / num_mnz_models))
            for sol in sols: print(sol)
            
            outputs = mnz_pred
            tot_time += tot_time_sample
            
        indices = torch.tensor([class_to_evaluate] * outputs.shape[0])
        if use_cuda:
            indices = indices.cuda()

        # focus only on the given subset of classes
        filtered_outputs = torch.gather(outputs, 1, indices)
        filtered_labels = torch.gather(labels_clip, 1, indices)
        
        save_filtered_outputs(str(sample_test), eval_type, filtered_outputs, filtered_labels, filtered_outputs_to_save)
        
        filtered_outputs = filtered_outputs.data.numpy()
        filtered_labels = filtered_labels.cpu().data.numpy()
        
        assert len(filtered_outputs) == len(filtered_labels)
        predictions.extend(filtered_outputs)
        ground_truth.extend(filtered_labels)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    # compute metrics
    avg_precision_score = average_precision_score(ground_truth, predictions, average=None)
    predictions = (predictions > f1_threshold).astype(int)
    ground_truth = (ground_truth > f1_threshold).astype(int)
    results_actions = precision_recall_fscore_support(ground_truth, predictions, average=None)
    f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]

    if eval_type == "nmnz":
        print("\n\n Tot time minizinc calls {:.2f}".format(tot_time))
        
    print('\n\nEpoch: %d, F1-Score: %s' % (epoch, str(f1_scores)), flush=True)
    print('Epoch: %d, Average Precision: %s' % (epoch, str(avg_precision_score)), flush=True)
    print('Epoch: %d, F1-Score: %4f, mAP: %4f'
          % (epoch, np.nanmean(f1_scores), np.nanmean(avg_precision_score)),
          flush=True)

    # save nn output in a pickle
    if not os.path.exists(path_to_nn_output_file):
        with open(path_to_nn_output_file, "wb") as pf:
            pickle.dump(nn_output, pf, protocol=pickle.HIGHEST_PROTOCOL)
            
    with open(path_to_filtered_outputs, "wb") as ffo:
        pickle.dump(filtered_outputs_to_save, ffo, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation on test set")
    
    parser.add_argument("--eval_type", type=str, help="Specify: n (Neural) -- nmnz (Neural + Minizinc)")
    parser.add_argument("--path_to_model", type=str, help="Path to the mlad pretrained model")
    parser.add_argument("--path_to_conf", type=str, help="Path to configuration file")
    parser.add_argument("--path_to_mnz", type=str, help="Path to minizinc models")
    parser.add_argument("--path_to_data", type=str, help="Path to all se files")
    parser.add_argument("--path_to_filtered_data", type=str, help="Path to filtered se files")
    parser.add_argument("--path_to_ann", type=str, help="Path to annotation json")
    parser.add_argument("--path_to_nn_output", type=str, help="Path to the pre-computed nn output")
    parser.add_argument("--criteria", type=str, help="criteria to select se for a clip (a=avg)")
    args = parser.parse_args()
    
    dataset = "multithumos"
    mode = "test"
    eval_type = args.eval_type
    path_to_model = args.path_to_model
    path_to_conf = args.path_to_conf
    path_to_mnz = args.path_to_mnz
    path_to_data = args.path_to_data
    path_to_filtered_data = args.path_to_filtered_data
    path_to_annotations_json = args.path_to_ann
    path_to_nn_output = args.path_to_nn_output
    criteria = args.criteria
    
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
    num_clips = cfg_model["num_clips"]
    num_mlad_layers = cfg_model["num_mlad_layers"]
    dim_of_features = cfg_model["dim_of_features"]
    # evaluation is done only to a subset of classes (depending on the events we manage)
    cfg_model["class_to_evaluate"] = [dataset_classes[class_name] - 1 for class_name in cfg_model["class_to_evaluate"]]
    cfg_model["path_to_nn_output"] = path_to_nn_output
    # name of the folder of the mnz model to evaluate
    cfg_model["mnz_models"] = path_to_mnz.split("/")[-2]

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
    
    # load mnz model for each structured event (se)
    mnz_files_names = os.listdir(path_to_mnz)
    mnz_models = {}
    for mnz_file_name in mnz_files_names:
        se_name = mnz_file_name.split(".")[0]
        with open(path_to_mnz + mnz_file_name, "r") as mzn_file:
            mnz_models[se_name] = mzn_file.read()
    
    if use_cuda:
        nn_model.cuda()
    
    # load features and se test events
    features_test = h5.File(cfg_dataset.combined_test_file, 'r')
    se_test = load_data(mode, path_to_data, path_to_filtered_data, path_to_annotations_json, features_test)
    
    # epoch of the model to load
    epoch = 50
    evaluate(eval_type, cfg_model, cfg_dataset, epoch, nn_model, features_test, se_test, mnz_models, criteria)
