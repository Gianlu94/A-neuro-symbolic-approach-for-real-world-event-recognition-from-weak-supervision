import os
import time

import numpy as np
import random
import json
import torch
from torch.autograd.variable import Variable
from torch import nn
from tqdm import tqdm
import pymzn
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
from tensorboardX import SummaryWriter

from utils import convert_to_float_tensor, get_avg_actions_durations_in_f
from minizinc.my_functions import build_problem, fill_mnz_pred, get_best_sol


def build_labels(video_id, annotations_file, num_features, num_classes, add_background=False):
    annotations = json.load(open(annotations_file, 'r'))
    labels = np.zeros((num_features, num_classes), np.float32)
    fps = num_features/annotations[video_id]['duration']
    for annotation in annotations[video_id]['actions']:
        for fr in range(0, num_features, 1):
            if fr/fps >= annotation[1] and fr/fps <= annotation[2]:
                labels[fr, annotation[0] - 1] = 1    # will make the first class to be the last for datasets other than Multi-Thumos #
    if add_background == True:
        new_labels = np.zeros((num_features, num_classes + 1))
        for i, label in enumerate(labels):
            new_labels[i,0:-1] = label
            if np.max(label) == 0:
                new_labels[i,-1] = 1
        return new_labels
    return labels


def build_dataset(se_list, features, cfg_dataset):
    dataset = {}
    for i, sample in enumerate(se_list):
        video, interval_cut_f = sample[0], sample[4]
        
        features_video = features[video]
        features_clip = features_video[interval_cut_f[0]:interval_cut_f[1]+1]
        labels_clip = build_labels(video, cfg_dataset.annotations_file, len(features_video), cfg_dataset.num_classes, False)
        
        dataset.setdefault(video, [features_clip, labels_clip])
    
    return dataset


def evaluate(epoch, se_list, features, nn_model, num_clips, class_to_evaluate, f1_threshold, cfg_dataset, use_cuda):
    nn_model.eval()
    num_se = len(se_list)
    predictions = []
    ground_truth = []
    print("\nStarting evaluation")
    start_time = time.time()
    for i, sample in enumerate(se_list):
        video, duration, num_features, se_name, interval_cut_f, event, _ = sample
    
        begin_se_c = event[0] - interval_cut_f[0]
        end_se_c = begin_se_c + event[1] - event[0]

        print("\nProcessing sample [{}, {}, {}]  {}/{}  ".format(video, se_name, (begin_se_c, end_se_c), i + 1, num_se), end="")

        # get features for the current video
        features_video = np.array(features[video])
        features_video = Variable(torch.from_numpy(features_video).type(torch.FloatTensor))

        # get labels
        labels_video = build_labels(
            video, cfg_dataset.annotations_file, len(features_video), cfg_dataset.num_classes, False)

        labels_video = Variable(torch.from_numpy(labels_video).type(torch.FloatTensor))

        # get clip and its labels
        features_clip = features_video[interval_cut_f[0]:interval_cut_f[1] + 1]
        labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
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

            outputs = outputs.reshape(-1, 65)

            indices = torch.tensor([class_to_evaluate] * outputs.shape[0])
            if use_cuda:
                indices = indices.cuda()

            # focus only on the given subset of classes
            filtered_outputs = torch.gather(outputs, 1, indices)
            filtered_labels = torch.gather(labels_clip, 1, indices)

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
    end_time = time.time()
    print("\n\nTIME: {:.2f}".format(end_time - start_time))
    print('Epoch: %d, Precision per class: %s' % (epoch, precision), flush=True)
    print('Epoch: %d, Recall per class: %s' % (epoch, recall), flush=True)
    print('Epoch: %d, F1-Score per class: %s\n' % (epoch, str(f1_scores)), flush=True)
    print('Epoch: %d, Average Precision: %s' % (epoch, str(avg_precision_score)), flush=True)
    print('Epoch: %d, F1-Score: %4f, mAP: %4f'
          % (epoch, np.nanmean(f1_scores), np.nanmean(avg_precision_score)),
          flush=True)


def train_model(se_train, se_test, features_train, features_test, nn_model, mnz_models, cfg_model, cfg_dataset, dataset_classes):
    # training info
    run_id = cfg_model["run_id"]
    use_cuda = cfg_model["use_cuda"]
    num_epochs = cfg_model["num_epochs"]
    save_epochs = cfg_model["save_epochs"]
    batch_size = cfg_model["batch_size"]
    num_batches = len(se_train) // batch_size
    learning_rate = cfg_model["learning_rate"]
    weight_decay = cfg_model["weight_decay"]
    optimizer = cfg_model["optimizer"]
    f1_threshold = cfg_model["f1_threshold"]
    num_clips = cfg_model["num_clips"]
    class_to_evaluate = [dataset_classes[class_name] - 1 for class_name in cfg_model["class_to_evaluate"]]
    avg_actions_durations_s = cfg_model["avg_actions_durations_s"]
    
    saved_models_dir = cfg_dataset.saved_models_dir
    # signature
    train_info = "/{}/ne_{}_bs_{}_lr_{}_wd_{}_opt_{}_f1th{}/".format(run_id,
                                                                     num_epochs, batch_size, learning_rate,
                                                                     weight_decay, optimizer, f1_threshold)
    saved_models_dir += train_info
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # to save metrics during training
    writer = SummaryWriter(cfg_dataset.tf_logs_dir + train_info)

    num_mnz_models = len(mnz_models.keys())
    bceWLL = nn.BCEWithLogitsLoss()
    
    if optimizer == "Adam":
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    features_train = convert_to_float_tensor(features_train)
    features_test = convert_to_float_tensor(features_test)
    
    se_train += se_test
    num_training_examples = len(se_train)
    # print("c")
    # import time
    # start_time = time.time()
    # data_train = build_dataset(se_train, features_train, cfg_dataset)
    # data_test = build_dataset(se_test, features_test, cfg_dataset)
    # end_time = time.time()
    # print(end_time-start_time)
    # breakpoint()
    #evaluate(0, se_test, features_test, nn_model, num_clips, class_to_evaluate, f1_threshold, cfg_dataset, use_cuda)
    #breakpoint()
    optimizer.zero_grad()
    for epoch in range(1, num_epochs+1):
        print("--- START EPOCH {}".format(epoch))
        nn_model.train()
        random.shuffle(se_train)

        start_time_epoch = time.time()
        tot_time_mnz = 0.
        epoch_loss = 0.
        batch_loss = 0.
        
        for index, sample_train in enumerate(se_train):
            print(sample_train)

            # get video, se name and interval where the se is happening
            video, duration, num_features, se_name, interval_cut_f = \
                sample_train[0], sample_train[1], sample_train[2], sample_train[3], sample_train[4]

            # get features for the current video
            if video in features_train:
                features_video = features_train[video]
            elif video in features_test:
                features_video = features_test[video]
            
            # get clip and its labels
            features_clip = features_video[interval_cut_f[0]:interval_cut_f[1] + 1]
            # if len(features_clip) < num_clips:
            #     # padding
            #     features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
            #     features_clip = torch.cat((features_clip, features_to_append), 0)
            
            #labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
            # get the output from the network
            out = nn_model(features_clip.unsqueeze(0))
            outputs = out['final_output'][0]

            if video in features_train:
                # get labels
                labels_video = build_labels(
                    video, cfg_dataset.annotations_file, len(features_video), cfg_dataset.num_classes, False)
                labels_clip = torch.tensor(labels_video[interval_cut_f[0]:interval_cut_f[1] + 1])
                sample_loss = bceWLL(outputs, labels_clip)
            else:
                # mnz
                outputs_transpose = outputs.transpose(0, 1)
                
                # minizinc part
                
                tot_time_sample = 0
                sols = []
                for se_name, mnz_model in mnz_models.items():
                    
                    avg_actions_durations_f = get_avg_actions_durations_in_f(se_name, duration, num_features, avg_actions_durations_s)
                    mnz_problem, _ = build_problem(se_name, mnz_model, nn.Sigmoid()(outputs_transpose), dataset_classes,
                                                   avg_actions_durations_f)
                    start_time = time.time()
                    sol = pymzn.minizinc(mnz_problem, solver=pymzn.Chuffed())
                    end_time = time.time()
        
                    tot_time_sample += end_time - start_time
                    sols.append(sol)
            
                # get best solution
                mnz_pred = torch.zeros(outputs.shape)
                best_sol, se_name, se_interval = get_best_sol(sols, "max_avg", outputs, dataset_classes)
                class_to_evaluate_mnz = fill_mnz_pred(mnz_pred, best_sol, se_name, dataset_classes)
    
                print("--- ({} calls to mnz) -- tot_time = {:.2f} - avg_time = {:.2f} \n".format(
                    num_mnz_models, tot_time_sample, tot_time_sample / num_mnz_models))
                
                for sol in sols: print(sol)
                
                # outputs = mnz_pred
                tot_time_mnz += tot_time_sample
    
                indices = torch.tensor([class_to_evaluate_mnz] * outputs.shape[0])
                if use_cuda:
                    indices = indices.cuda()
    
                # focus only on the given subset of classes
                filtered_outputs = torch.gather(outputs, 1, indices)
                filtered_mnz_pred = torch.gather(mnz_pred, 1, indices)
                
                #filtered_labels = torch.gather(labels_clip, 1, indices)
                
                sample_loss = (
                        bceWLL(filtered_outputs[se_interval[0]:se_interval[1]+1, :3], filtered_mnz_pred[se_interval[0]:se_interval[1]+1, :3]) +
                        bceWLL(filtered_outputs[:, 3], filtered_mnz_pred[:, 3])) / 2
            
            batch_loss += sample_loss

            sample_loss.backward()
            print("\nEpoch {} - sample {}/{} ---- loss = {}".format(epoch, index+1,  num_training_examples, sample_loss))
            # batch update
            if (index+1) % batch_size == 0:
                print("\nEpoch {} BATCH ---- loss = {}\n".format(epoch, batch_loss))
                epoch_loss += batch_loss / num_batches
                batch_loss = 0.
                optimizer.step()
                optimizer.zero_grad()
        
        end_time_epoch = time.time()
        print("--- END EPOCH {} -- LOSS {} -- TIME {:.2f} -- TIME MNZ {:.2f}\n".format(
            epoch, epoch_loss/num_training_examples, end_time_epoch-start_time_epoch, tot_time_mnz ))
        writer.add_scalar("Training loss", epoch_loss/num_training_examples, epoch)
        
        # TODO: save the model (criteria)
        if epoch % save_epochs == 0 or epoch == 1:
            state = {
                "epoch": epoch,
                "state_dict": nn_model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, saved_models_dir + "model_{}_loss_{:.4f}.pth".format(epoch, epoch_loss))
        
        # TODO: for now evaluation is done after each epoch, change it
        evaluate(epoch, se_test, features_test, nn_model, num_clips, class_to_evaluate, f1_threshold, cfg_dataset, use_cuda)
            
            
            
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
