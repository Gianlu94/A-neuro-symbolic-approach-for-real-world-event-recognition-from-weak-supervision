import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import torch
from torch.autograd.variable import Variable
from torch import nn
from tqdm import tqdm
import pymzn
from sklearn.metrics import (
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support)
from tensorboardX import SummaryWriter

from utils import convert_to_float_tensor, get_avg_actions_durations_in_f, get_textual_label_from_tensor
from minizinc.my_functions import build_problem_exp1, fill_mnz_pred_exp1, get_best_sol

data_dec = {}


def build_labels(video_id, annotations_file, num_features, num_classes, add_background=False):
    annotations = json.load(open(annotations_file, 'r'))
    labels = np.zeros((num_features, num_classes), np.float32)
    fps = num_features / annotations[video_id]['duration']
    for annotation in annotations[video_id]['actions']:
        for fr in range(0, num_features, 1):
            if fr / fps >= annotation[1] and fr / fps <= annotation[2]:
                labels[fr, annotation[
                    0] - 1] = 1  # will make the first class to be the last for datasets other than Multi-Thumos #
    if add_background == True:
        new_labels = np.zeros((num_features, num_classes + 1))
        for i, label in enumerate(labels):
            new_labels[i, 0:-1] = label
            if np.max(label) == 0:
                new_labels[i, -1] = 1
        return new_labels
    return labels


def evaluate(
        epoch, mode, se_list, features, labels, labels_textual, nn_model, loss, num_clips, f1_threshold, mnz_models,
        se_labels, avg_actions_durations_s, use_cuda, classes_names, writer, brief_summary, epochs_predictions
):
    
    nn_model.eval()
    se_names = list(se_labels.keys())
    num_se = len(se_list)
    num_mnz_models = len(mnz_models.keys())
    actions_predictions = []
    actions_ground_truth = []
    
    tot_loss = 0.
    print("\nStarting evaluation")
    start_time_ev = time.time()
    tot_time_mnz = 0

    for i, example in enumerate(se_list):
        video, gt_se_name, duration, num_features, se_interval, _ = example

        new_begin_se = 0
        new_end_se = se_interval[1] - se_interval[0]

        print("\nProcessing example [{}, {}, {}]  {}/{}  ".format(video, gt_se_name, (se_interval), i + 1, num_se),
              end="")

        # get features for the current video
        features_video = np.array(features[video])
        features_video = Variable(torch.from_numpy(features_video).type(torch.FloatTensor))
        
        example_id = "{}-{}-{}".format(video, gt_se_name, se_interval)

        labels_clip = labels["{}-{}-{}".format(video, gt_se_name, se_interval)]
        labels_clip_textual = labels_textual[example_id]
        
        #labels_video = Variable(torch.from_numpy(labels_video).type(torch.FloatTensor))
        
        # get clip and its labels
        features_clip = features_video[se_interval[0]:se_interval[1] + 1]
        #labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
        with torch.no_grad():
            if num_clips > 0:
                if len(features_clip) < num_clips:
                    # padding
                    features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
                    #labels_to_append = torch.zeros(num_clips - len(labels_clip) % num_clips, labels_clip.shape[1])
                    features_clip = torch.cat((features_clip, features_to_append), 0)
                    #labels_clip = torch.cat((labels_clip, labels_to_append), 0)
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
            outputs = outputs.squeeze(0)
            outputs = outputs[new_begin_se:new_end_se + 1]
            # mnz
            outputs_transpose = outputs.transpose(0, 1)

            # minizinc part
            tot_time_example = 0
            sols = []
            
            for se_name, mnz_model in mnz_models.items():
                avg_actions_durations_f = get_avg_actions_durations_in_f(se_name, duration, num_features,
                                                                         avg_actions_durations_s)
                mnz_problem, _ = build_problem_exp1(se_name, mnz_model, nn.Sigmoid()(outputs_transpose), avg_actions_durations_f)
                start_time = time.time()
    
                sol = pymzn.minizinc(mnz_problem, solver=pymzn.gurobi)
                end_time = time.time()
                tot_time_example += end_time - start_time
                sols.append(sol)
        
            tot_time_mnz += tot_time_example

            # get best solution
            mnz_pred = torch.zeros(outputs.shape)
            best_sol, predicted_se_name, _ = get_best_sol(sols, "max_avg", nn.Sigmoid()(outputs), classes_names)
            fill_mnz_pred_exp1(mnz_pred, best_sol, predicted_se_name)

            example_loss = loss((mnz_pred * outputs)[new_begin_se:new_end_se+1], labels_clip[new_begin_se:new_end_se+1]) / labels_clip.shape[1]
            tot_loss += example_loss
            print("--- ({} calls to mnz) -- tot_time = {:.2f} - avg_time = {:.2f} \n".format(
                num_mnz_models, tot_time_example, tot_time_example / num_mnz_models))

            for sol in sols: print(sol)
            print("\n best_sol {}".format(predicted_se_name))
            print("Ground Thruth: {}".format(labels_clip_textual))

            outputs = mnz_pred.reshape(-1, len(classes_names))

            outputs = outputs.data.numpy()
            labels_clip = labels_clip.cpu().data.numpy()
            assert len(outputs) == len(labels_clip)

            epochs_predictions["epoch"].append(epoch)
            epochs_predictions["video"].append(video)
            epochs_predictions["gt_se_names"].append(gt_se_name)
            epochs_predictions["pred_se_names"].append(predicted_se_name)
            epochs_predictions["se_interval"].append(se_interval)
            epochs_predictions["ground_truth"].append(labels_clip[new_begin_se: new_end_se + 1])
            epochs_predictions["predictions"].append(mnz_pred[new_begin_se: new_end_se + 1])
            
            se_predictions = np.zeros((outputs.shape[0], num_mnz_models))
            se_predictions[:, se_labels[predicted_se_name]] = 1
            se_gt = np.zeros((outputs.shape[0], num_mnz_models))
            se_gt[:, se_labels[gt_se_name]] = 1

            actions_predictions.extend(np.concatenate((outputs, se_predictions), axis=1))
            actions_ground_truth.extend(np.concatenate((labels_clip, se_gt), axis=1))

    actions_ground_truth = np.array(actions_ground_truth)
    actions_predictions = np.array(actions_predictions)
 
    # compute metrics
    actions_avg_precision_score = average_precision_score(actions_ground_truth, actions_predictions, average=None)

    # cf_matrix = confusion_matrix(np.argmax(ground_truth, 1), np.argmax(predictions, 1))
    # cf_matrix_to_display = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=classes_names)
    # cf_matrix_to_display.plot(xticks_rotation="vertical", cmap=plt.cm.Blues, values_format='g')

    actions_results = precision_recall_fscore_support(actions_ground_truth, actions_predictions, average=None)
    
    actions_f1_scores, actions_precision, actions_recall = actions_results[2], actions_results[0], actions_results[1]

    
    end_time_ev = time.time()

    metrics_to_print = """
        \nTIME: {:.2f} - time MNZ {:.2f}
        {} -- Epoch: {}, Loss: {}
        {} -- Epoch: {}, Precision per class: {}
        {} -- Epoch: {}, Recall per class: {}
        {} -- Epoch: {}, F1-Score per class: {}
        {} -- Epoch: {}, Average Precision: {}
        {} -- Epoch: {}, F1-Score: {:.4f}, mAP: {:.4f}
    """.format(
        end_time_ev-start_time_ev, tot_time_mnz,
        mode, epoch, tot_loss.item()/num_se,
        mode, epoch, actions_precision,
        mode, epoch, actions_recall,
        mode, epoch, str(actions_f1_scores),
        mode, epoch, str(actions_avg_precision_score),
        mode, epoch, np.nanmean(actions_f1_scores), np.nanmean(actions_avg_precision_score)
    )
    
    print(metrics_to_print, flush=True)
    brief_summary.write(metrics_to_print)

    if writer is not None:
        for i, class_name in enumerate(classes_names + se_names):
            #writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("F1 Score {} ".format(class_name), actions_f1_scores[i], epoch)
            writer.add_scalar("Precision {} ".format(class_name), actions_precision[i], epoch)
            writer.add_scalar("Recall {} ".format(class_name), actions_recall[i], epoch)
            #writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)
        
        writer.add_scalar('Loss', tot_loss.item()/num_se, epoch)
        writer.add_scalar('Avg F1 Score', np.nanmean(actions_f1_scores), epoch)
        writer.add_scalar('Avg Precision', np.nanmean(actions_precision), epoch)
        writer.add_scalar('Avg Recall', np.nanmean(actions_recall), epoch)
        writer.add_scalar('Avg AP', np.nanmean(actions_avg_precision_score), epoch)

    return np.nanmean(actions_avg_precision_score) #, cf_matrix_to_display


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
            if begin_ev == 0:
                dec_se["begin_f_run"] = se_interval[0]
                
            action_duration = dec_se["end_f_run"].values[0] - dec_se["begin_f_run"].values[0]
            
            intervals_events.append([begin_ev, begin_ev + action_duration])

            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])

            begin_ev = dec_se["begin_f_fall"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + _set_least_value(dec_se["end_f_fall"].values[0], se_interval[1]) - dec_se["begin_f_fall"].values[0]])
            
            labels_indices = list(range(0, 3))
        elif se_name == "HammerThrow":
            # get decomposition of the current se
            dec_se = data_dec_se[
                (data_dec_se["video"] == video) & (data_dec_se["begin_f_ht"] == se_interval[0]) & (
                        data_dec_se["end_f_ht"] == se_interval[1])].copy(deep=True)

            begin_ev = _set_nn_value(dec_se["begin_f_ht_wu"].values[0] - se_interval[0])
            
            if begin_ev == 0:
                dec_se["begin_f_ht_wu_run"] = se_interval[0]
            
            action_duration = dec_se["end_f_ht_wu"].values[0] - dec_se["begin_f_ht_wu"].values[0]

            intervals_events.append([begin_ev, begin_ev + action_duration])

            begin_ev = dec_se["begin_f"].values[0] - se_interval[0]
            intervals_events.append([begin_ev, begin_ev + dec_se["end_f"].values[0] - dec_se["begin_f"].values[0]])

            begin_ev = dec_se["begin_f_ht_r"].values[0] - se_interval[0]
            intervals_events.append(
                [begin_ev, begin_ev + _set_least_value(dec_se["end_f_ht_r"].values[0], se_interval[1]) - dec_se["begin_f_ht_r"].values[0]])

            labels_indices = list(range(3, 6))

        rows = []
        columns = []
     
        for i in labels_indices:
            begin, end = int(intervals_events[0][0]), int(intervals_events[0][1])
            rows += [i] * (end-begin+1)
            columns += [j for j in range(begin, end+1)]
            intervals_events.pop(0)
        
        label_key = "{}-{}-{}".format(video, se_name, se_interval)
        dec_labels[label_key] = torch.zeros((cfg_train["classes"], se_interval[1]-se_interval[0]+1))
        
        dec_labels[label_key][rows, columns] = 1
        dec_labels[label_key] = dec_labels[label_key].transpose(0, 1)

    return dec_labels


def train_exp1_mnz(se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset, mnz_models):
    
    # training info
    run_id = cfg_train["run_id"]
    use_cuda = cfg_train["use_cuda"]
    num_epochs = cfg_train["num_epochs"]
    save_epochs = cfg_train["save_epochs"]
    batch_size = cfg_train["batch_size"]
    num_batches = len(se_train) // batch_size
    learning_rate = cfg_train["learning_rate"]
    weight_decay = cfg_train["weight_decay"]
    optimizer = cfg_train["optimizer"]
    f1_threshold = cfg_train["f1_threshold"]
    num_clips = cfg_train["num_clips"]
    classes_names = cfg_train["classes_names"]
    avg_actions_durations_s = cfg_train["avg_actions_durations_s"]
    structured_events = cfg_train["structured_events"]
    
    saved_models_dir = cfg_dataset.saved_models_dir
    # signature
    train_info = "/{}/ne_{}_bs_{}_lr_{}_wd_{}_opt_{}_f1th{}/".format(run_id,
                                                                     num_epochs, batch_size, learning_rate,
                                                                     weight_decay, optimizer, f1_threshold)
    saved_models_dir += train_info
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # path_to_cf = cfg_dataset.tf_logs_dir + train_info + "confusion_matrix/"
    # for split in ["train", "val", "test"]:
    #     os.makedirs(path_to_cf + split + "/", exist_ok=True)

    epochs_predictions = {
        "train":
            {
                "epoch": [], "video": [], "gt_se_names": [],  "pred_se_names": [], "se_interval": [], "ground_truth": [],
                "predictions": []
            },
        "val":
            {
                "epoch": [], "video": [], "gt_se_names": [],  "pred_se_names": [], "se_interval": [], "ground_truth": [],
                "predictions": []
            },
        "test":
            {
                "epoch": [], "video": [], "gt_se_names": [],  "pred_se_names": [], "se_interval": [], "ground_truth": [],
                "predictions": []
            }
    }

    # to save metrics during training
    writer_train = SummaryWriter(cfg_dataset.tf_logs_dir + train_info + "train/")
    writer_val = SummaryWriter(cfg_dataset.tf_logs_dir + train_info + "val/")
    brief_summary = open("{}/brief_summary.txt".format(cfg_dataset.tf_logs_dir + train_info), "w")
    best_model_ep = 0

    bceWLL = nn.BCEWithLogitsLoss()
    
    if optimizer == "Adam":
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    features_train = convert_to_float_tensor(features_train)
    features_test = convert_to_float_tensor(features_test)

    # se_train = se_train[:5]
    # se_val = [se_val[1]] + [se_val[-1]]
    # se_test = [se_test[1]] + [se_test[-1]]
    
    labels_train = get_labels(se_train, cfg_train)
    labels_train_textual = get_textual_label_from_tensor(labels_train)
    labels_val = get_labels(se_val, cfg_train)
    labels_val_textual = get_textual_label_from_tensor(labels_val)
    labels_test = get_labels(se_test, cfg_train)
    labels_test_textual = get_textual_label_from_tensor(labels_test)
    
    max_fmap_score = 0.
    
    num_training_examples = len(se_train)
    
    # fmap_score = evaluate(
    #     24, "Train", se_train, features_train, labels_train_dec, nn_model, bceWLL, num_clips, f1_threshold,
    #     mnz_models, avg_actions_durations_s, use_cuda, classes_names, None, brief_summary
    # )
    
    optimizer.zero_grad()

    for epoch in range(1, num_epochs + 1):
        start_time_epoch = time.time()
        print("\n--- START EPOCH {}\n".format(epoch))
        nn_model.train()
        random.shuffle(se_train)

        tot_time_mnz = 0.
        epoch_loss = 0.
        batch_loss = 0.
        
        for index, example_train in enumerate(se_train):
            print(example_train)
            # get video, duration, num_features, se name and interval where the se is happening
            video, se_name, duration, num_features, se_interval = \
                example_train[0], example_train[1], example_train[2], example_train[3], example_train[4]
            
            new_begin_se = 0
            new_end_se = se_interval[1] - se_interval[0]
            
            features_video = features_train[video]
            
            # get clip and its labels
            features_clip = features_video[se_interval[0]:se_interval[1] + 1]
            example_id = "{}-{}-{}".format(video, se_name, se_interval)
            labels_clip_textual = labels_train_textual[example_id]
            
            # if len(features_clip) < num_clips:
            #     # padding
            #     features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
            #     features_clip = torch.cat((features_clip, features_to_append), 0)
            
            # labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
            # get the output from the network
            out = nn_model(features_clip.unsqueeze(0))
            
            outputs = out['final_output'][0]

            # mnz
            outputs_transpose = outputs.transpose(0, 1)

            # minizinc part
            tot_time_example = 0
            
            avg_actions_durations_f = get_avg_actions_durations_in_f(se_name, duration, num_features, avg_actions_durations_s)
            
            mnz_problem, _ = build_problem_exp1(se_name, mnz_models[se_name], nn.Sigmoid()(outputs_transpose), avg_actions_durations_f)
            #if se_name == "HammerThrow": breakpoint()
            start_time = time.time()
            sol = pymzn.minizinc(mnz_problem, solver=pymzn.gurobi)
            end_time = time.time()
            tot_time_example += end_time - start_time

            tot_time_mnz += tot_time_example

            mnz_pred = torch.zeros(outputs.shape)

            fill_mnz_pred_exp1(mnz_pred, sol, se_name)
            
            print("--- call to mnz - time = {:.2f}\n".format(tot_time_example))

            example_loss = bceWLL(outputs[new_begin_se:new_end_se+1], mnz_pred[new_begin_se:new_end_se+1])
            batch_loss += example_loss
            
            example_loss.backward()
            
            print("MNZ sol: {}".format(str(sol)))
            print("Ground Thruth: {}".format(labels_clip_textual))
            
            print(
                "\nEpoch {} - example {}/{} ---- loss = {:.4f}\n".format(epoch, index + 1, num_training_examples, example_loss))
            
            # batch update
            if (index + 1) % batch_size == 0:
                print("\nEpoch {} BATCH ---- loss = {}\n".format(epoch, batch_loss))
                epoch_loss += batch_loss / num_batches
                batch_loss = 0.
                optimizer.step()
                optimizer.zero_grad()

            labels_clip = labels_train[example_id]
            epochs_predictions["train"]["epoch"].append(epoch)
            epochs_predictions["train"]["video"].append(video)
            epochs_predictions["train"]["gt_se_names"].append(se_name)
            epochs_predictions["train"]["se_interval"].append(se_interval)
            epochs_predictions["train"]["ground_truth"].append(labels_clip[new_begin_se: new_end_se+1].cpu().detach().numpy())
            epochs_predictions["train"]["predictions"].append(mnz_pred[new_begin_se: new_end_se+1].cpu().detach().numpy())

        end_time_epoch = time.time()
        print("--- END EPOCH {} -- LOSS {} -- TIME {:.2f}-- TIME MNZ {:.2f}\n".format(
            epoch, epoch_loss, end_time_epoch - start_time_epoch, tot_time_mnz))
        
        writer_train.add_scalar("Loss", epoch_loss / num_training_examples, epoch)
        
        # TODO: for now evaluation is done after each epoch, change it
        # evaluate(
        #     epoch, "Train", se_train, features_train, nn_model, num_clips, original_classes, f1_threshold, cfg_dataset, use_cuda,
        #     cfg_train["range_classes_to_train"], classes_names, writer_train, brief_summary
        # )
        # plt.savefig(path_to_cf + "train/cf_epoch_{}.png".format(epoch))
        fmap_score = evaluate(
            epoch, "Validation", se_val, features_train, labels_val, labels_val_textual, nn_model, bceWLL, num_clips, f1_threshold, mnz_models,
            structured_events, avg_actions_durations_s, use_cuda, classes_names, writer_val, brief_summary, epochs_predictions["val"]
        )
        #plt.savefig(path_to_cf + "val/cf_epoch_{}.png".format(epoch))
        
        if fmap_score > max_fmap_score:
            best_model_ep = epoch
            max_fmap_score = fmap_score
            
        #     for f in os.listdir(saved_models_dir):
        #         os.remove(os.path.join(saved_models_dir, f))

        state = {
            "epoch": epoch,
            "state_dict": nn_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(state, saved_models_dir + "model_{}_loss_{:.4f}.pth".format(epoch, epoch_loss))
    
    best_model_path = os.path.join(saved_models_dir, os.listdir(saved_models_dir)[0])
    
    # load and evaluate best model on test set
    if use_cuda:
        state = torch.load(best_model_path)
    else:
        state = torch.load(best_model_path, map_location=torch.device('cpu'))

    nn_model.load_state_dict(state["state_dict"])

    fmap_score = evaluate(
        best_model_ep, "Test", se_test, features_test, labels_test, labels_test_textual, nn_model, bceWLL, num_clips, f1_threshold, mnz_models,
        structured_events, avg_actions_durations_s, use_cuda, classes_names, None, brief_summary, epochs_predictions["test"]
    )

    with open("{}/epochs_predictions.pickle".format(cfg_dataset.tf_logs_dir + train_info), "wb") as epp_file:
        pickle.dump(epochs_predictions, epp_file, protocol=pickle.HIGHEST_PROTOCOL)

    #plt.savefig(path_to_cf + "test/cf_best_model.png")
    brief_summary.close()
    print(fmap_score)
    print(best_model_ep)