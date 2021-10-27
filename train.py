import os

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

from utils import convert_to_float_tensor, convert_indices
from minizinc.my_functions import build_problem


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


def evaluate(cfg_model, cfg_dataset, class_to_evaluate, f1_threshold, epoch, nn_model, features_test, video_list, writer=None):
    nn_model.eval()
    
    num_clips = cfg_model["num_clips"]
    eval_mode = cfg_model["eval_mode"]
    
    use_cuda = cfg_model["use_cuda"]
    
    predictions, ground_truth = [], []

    for i, video in tqdm(enumerate(video_list)):
        features_video = features_test[video]
        labels = build_labels(
            video, cfg_dataset.annotations_file, len(features_video), cfg_dataset.num_classes, False)
        
        features = np.array(features_video)
        labels = np.array(labels)
        features = Variable(torch.from_numpy(features).type(torch.FloatTensor))
        labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
        assert len(features) == len(labels)

        with torch.no_grad():
            if num_clips > 0:
                eval_mode = eval_mode
                if len(features) < num_clips:
                    eval_mode = 'pad'
                if eval_mode == 'truncate':
                    features = features[0:len(features) - (len(features) % num_clips)]
                    labels = labels[0:len(labels) - (len(labels) % num_clips)]
                    features = torch.stack(
                        [features[i:i + num_clips] for i in range(0, len(features), num_clips)])
                    labels = torch.stack([labels[i:i + num_clips] for i in range(0, len(labels), num_clips)])
                elif eval_mode == 'pad':
                    features_to_append = torch.zeros(num_clips - len(features) % num_clips, features.shape[1])
                    labels_to_append = torch.zeros(num_clips - len(labels) % num_clips, labels.shape[1])
                    features = torch.cat((features, features_to_append), 0)
                    labels = torch.cat((labels, labels_to_append), 0)
                    features = torch.stack(
                        [features[i:i + num_clips] for i in range(0, len(features), num_clips)])
                    labels = torch.stack([labels[i:i + num_clips] for i in range(0, len(labels), num_clips)])
                elif eval_mode == 'slide':
                    slide_rate = 16
                    features_to_append = torch.zeros(slide_rate - len(features) % slide_rate, features.shape[-1])
                    labels_to_append = torch.zeros(slide_rate - len(labels) % slide_rate, labels.shape[-1])
                    features = torch.cat((features, features_to_append), 0)
                    labels = torch.cat((labels, labels_to_append), 0)
                    features = torch.stack([features[i:i + num_clips] for i in
                                            range(0, len(features) - num_clips + 1, slide_rate)])
                    labels = torch.stack(
                        [labels[i:i + num_clips] for i in range(0, len(labels) - num_clips + 1, slide_rate)])
                assert len(features) > 0
            else:
                features = torch.unsqueeze(features, 0)
                labels = torch.unsqueeze(labels, 0)

            if use_cuda:
                features = features.cuda()
                labels = labels.cuda()

            out = nn_model(features)
            outputs = out['final_output']

            outputs = nn.Sigmoid()(outputs)

            outputs = outputs.reshape(-1, cfg_dataset.num_classes)
            labels = labels.reshape(-1, cfg_dataset.num_classes)
            
            indices = torch.tensor([class_to_evaluate] * outputs.shape[0])
            if use_cuda:
                indices.cuda()

            filtered_outputs = torch.gather(outputs, 1, indices)
            filtered_labels = torch.gather(labels, 1, indices)
            
            filtered_outputs = filtered_outputs.cpu().data.numpy()
            filtered_labels = filtered_labels.cpu().data.numpy()

        assert len(filtered_outputs) == len(filtered_labels)
        predictions.extend(filtered_outputs)
        ground_truth.extend(filtered_labels)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    avg_precision_score = average_precision_score(ground_truth, predictions, average=None)

    predictions = (np.array(predictions) > f1_threshold).astype(int)
    ground_truth = (np.array(ground_truth) > f1_threshold).astype(int)
    results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions), average=None)
    f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]

    print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores)), flush=True)
    print('Validation Epoch: %d, Average Precision: %s' % (epoch, str(avg_precision_score)), flush=True)
    print('Validation Epoch: %d, F1-Score: %4f, mAP: %4f'
           % (epoch, np.nanmean(f1_scores), np.nanmean(avg_precision_score)), flush=True)

    if writer is not None:
        writer.add_scalar('Validation F1 Score', np.nanmean(f1_scores), epoch)
        writer.add_scalar('Validation Precision', np.nanmean(precision), epoch)
        writer.add_scalar('Validation Recall', np.nanmean(recall), epoch)
        writer.add_scalar('Validation AP', np.nanmean(avg_precision_score), epoch)
    
    # return np.nanmean(avg_precision_score)
    
def _build_loss_for_the_network(sol, final_output, bce_loss):
    time_points = list(sol[0].values())
    max_time = final_output.shape[1] - 1
    
    loss = 0.
    
    for i in range(0, len(time_points), 2):
        begin = time_points[i] - 1
        end = time_points[i + 1] - 1
        
        index_atomic_action = i // 2
        
        # loss += (
        #     -torch.sum(final_output[index_atomic_action, begin:end+1])
        #     +torch.sum(final_output[index_atomic_action, :begin])
        #     +torch.sum(final_output[index_atomic_action, end:])
        # )
        loss += (
            bce_loss(final_output[index_atomic_action, begin:end + 1], torch.ones(end-begin+1)) +
            bce_loss(final_output[index_atomic_action, :begin], torch.zeros(begin-0)) +
            bce_loss(final_output[index_atomic_action, end+1:], torch.zeros(max_time-end))
        )
        
        return loss


def train_model(cfg_dataset, cfg_model, dataset_classes, se_train, features_train, features_test, nn_model, mnz_models):
    
    # training info
    run_id = cfg_model["run_id"]
    num_epochs = cfg_model["num_epochs"]
    save_epochs = cfg_model["save_epochs"]
    batch_size = cfg_model["batch_size"]
    num_batches = len(se_train) // batch_size

    learning_rate = cfg_model["learning_rate"]
    weight_decay = cfg_model["weight_decay"]
    optimizer = cfg_model["optimizer"]
    f1_threshold = cfg_model["f1_threshold"]
    class_to_evaluate = [dataset_classes[class_name] - 1 for class_name in cfg_model["class_to_evaluate"]]
    
    video_list_test = [line.rstrip().replace('.txt', '') for line in open(cfg_dataset.test_list, 'r').readlines()]
    
    saved_models_dir = cfg_dataset.saved_models_dir
    # signature
    train_info = "/{}/ne_{}_bs_{}_lr_{}_wd_{}_opt_{}_f1th{}/".format(run_id,
                                                                     num_epochs, batch_size, learning_rate,
                                                                     weight_decay, optimizer, f1_threshold)
    saved_models_dir += train_info
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # to save metrics during training
    writer = SummaryWriter(cfg_dataset.tf_logs_dir + train_info)
    
    num_samples = len(se_train)
    bce_loss = nn.BCELoss(reduction="sum")
    
    if optimizer == "Adam":
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    features_train = convert_to_float_tensor(features_train)
    
    for epoch in range(1, num_epochs+1):
        print("--- START EPOCH {}".format(epoch))
        nn_model.train()
        random.shuffle(se_train)

        epoch_loss = 0.
        batch_loss = 0.
        optimizer.zero_grad()
        for index, sample_train in enumerate(se_train):
            index += 1
            
            video, duration, se_name, begin_s, end_s = sample_train[0], sample_train[1], sample_train[2], sample_train[3], \
                                                    sample_train[4]
            # get features for the current video
            features_video = features_train[video]

            # convert from seconds to feature vectors
            begin_f, end_f = convert_indices(features_video.shape[0], duration, begin_s, end_s)
            
            # from indices get the clio
            features_se = features_video[begin_f:end_f + 1]
            
            final_output = torch.nn.Sigmoid()(nn_model(features_se.unsqueeze(0))["final_output"]).squeeze()
            
            final_output_transpose = final_output.transpose(0, 1)

            # get the model for the current se
            mnz_model = mnz_models[se_name]
            # build minizinc problem by including data
            mnz_problem = build_problem(se_name, mnz_model, final_output_transpose, dataset_classes)
            # get solutions+
            sol = pymzn.minizinc(mnz_problem)
            
            sample_loss = _build_loss_for_the_network(sol, final_output_transpose, bce_loss)
            batch_loss += sample_loss

            sample_loss.backward()
            print("Epoch {} - sample {}/{} ---- loss = {}".format(epoch, index, num_samples, sample_loss))

            # batch update
            if index % batch_size == 0:
                print("\nEpoch {} BATCH ---- loss = {}\n".format(epoch, batch_loss))
                epoch_loss += batch_loss / num_batches
                batch_loss = 0.
                optimizer.step()
                optimizer.zero_grad()
        
        print("--- END EPOCH {} -- LOSS {}\n".format(epoch, epoch_loss))
        writer.add_scalar("Training loss", epoch_loss, epoch)
        
        # TODO: save the model (criteria)
        if epoch % save_epochs == 0:
            state = {
                "epoch": epoch,
                "state_dict": nn_model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, saved_models_dir + "model_{}_loss_{:.4f}.pth".format(epoch, epoch_loss))
        
        # TODO: for now evaluation is done after each epoch, change it
        evaluate(cfg_model, cfg_dataset, class_to_evaluate, f1_threshold, epoch, nn_model, features_test, video_list_test, writer)
        
            
            
            
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
