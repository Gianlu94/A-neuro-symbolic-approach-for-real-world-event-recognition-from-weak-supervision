import os
import pickle
import random
import time

import numpy as np
import torch
from torch.autograd.variable import Variable
from torch import nn
import pymzn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tensorboardX import SummaryWriter

from dataset import get_labels, get_avg_labels
from utils import convert_to_float_tensor, get_textual_label_from_tensor
from minizinc.my_functions import build_problem_exp1, fill_mnz_pred_exp1, get_best_sol, set_prop_avg


def evaluate(
        epoch, mode, se_list, features, labels, labels_textual, nn_model, loss, ll_activation, selection_criteria,
        num_clips, mnz_models, se_labels, use_cuda, classes_names, classes_names_abb, writer,
        brief_summary, epochs_predictions
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
        
        # labels_video = Variable(torch.from_numpy(labels_video).type(torch.FloatTensor))
        
        # get clip and its labels
        features_clip = features_video[se_interval[0]:se_interval[1] + 1]
        # labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
        with torch.no_grad():
            if num_clips > 0:
                if len(features_clip) < num_clips:
                    # padding
                    features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
                    # labels_to_append = torch.zeros(num_clips - len(labels_clip) % num_clips, labels_clip.shape[1])
                    features_clip = torch.cat((features_clip, features_to_append), 0)
                    # labels_clip = torch.cat((labels_clip, labels_to_append), 0)
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
            outputs_transpose = ll_activation(outputs.transpose(0, 1))
            
            # minizinc part
            tot_time_example = 0
            sols = []
            
            # mnz_ground_truth
            mnz_gt = torch.zeros(outputs.shape)
            mnz_gt_sol = None
            
            for se_name, mnz_model in mnz_models.items():
                
                mnz_problem, _ = build_problem_exp1(se_name, mnz_model, outputs_transpose)
                
                start_time = time.time()
                
                sol = pymzn.minizinc(mnz_problem, solver=pymzn.gurobi)
                
                end_time = time.time()
                tot_time_example += end_time - start_time
                sols.append(sol)
                
                if se_name == gt_se_name and mode != "Test":
                    fill_mnz_pred_exp1(mnz_gt, sol, gt_se_name)
                    mnz_gt_sol = sol[0]
            
            tot_time_mnz += tot_time_example
            
            outputs_act = outputs_transpose.transpose(0, 1)
            
            # get best solution
            mnz_pred = torch.zeros(outputs.shape)
            
            best_sol, predicted_se_name, _ = get_best_sol(sols, selection_criteria, outputs, loss)
            
            fill_mnz_pred_exp1(mnz_pred, best_sol, predicted_se_name)
            
            if mode != "Test":
                labels_clip = mnz_gt
                labels_clip_textual = mnz_gt_sol

            # example_loss = loss(outputs, labels_clip)
            example_loss = loss(outputs, torch.argmax(labels_clip, 1))
            
            tot_loss += example_loss
            print("--- ({} calls to mnz) -- tot_time = {:.2f} - avg_time = {:.2f} \n".format(
                num_mnz_models, tot_time_example, tot_time_example / num_mnz_models))
            
            for sol in sols: print(sol)
            print("\n best_sol {}".format(predicted_se_name))
            print("Ground Thruth: {}".format(labels_clip_textual))
            
            outputs = mnz_pred.reshape(-1, len(classes_names))
            
            outputs = outputs.data.numpy()
            labels_clip = labels_clip.cpu().data.numpy()
            mnz_pred = mnz_pred.cpu().detach().numpy()
            
            assert len(outputs) == len(labels_clip)
            
            if epochs_predictions is not None:
                epochs_predictions["epoch"].append(epoch)
                epochs_predictions["video"].append(video)
                epochs_predictions["gt_se_names"].append(gt_se_name)
                epochs_predictions["pred_se_names"].append(predicted_se_name)
                epochs_predictions["se_interval"].append(se_interval)
                epochs_predictions["ground_truth"].append(labels_clip)
                epochs_predictions["outputs_act"].append(outputs_act.cpu().data.numpy())
                epochs_predictions["predictions"].append(mnz_pred)
            
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
        end_time_ev - start_time_ev, tot_time_mnz,
        mode, epoch, tot_loss.item() / num_se,
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
            # writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("F1 Score {} ".format(class_name), actions_f1_scores[i], epoch)
            writer.add_scalar("Precision {} ".format(class_name), actions_precision[i], epoch)
            writer.add_scalar("Recall {} ".format(class_name), actions_recall[i], epoch)
            # writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)
        
        writer.add_scalar('Loss', tot_loss.item() / num_se, epoch)
        writer.add_scalar('Avg F1 Score', np.nanmean(actions_f1_scores), epoch)
        writer.add_scalar('Avg Precision', np.nanmean(actions_precision), epoch)
        writer.add_scalar('Avg Recall', np.nanmean(actions_recall), epoch)
        writer.add_scalar('Avg AP', np.nanmean(actions_avg_precision_score), epoch)
    
    return np.nanmean(actions_avg_precision_score)


def train_exp2_mnz(se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset,
                   mnz_models):
    # training info
    run_id = cfg_train["run_id"]
    use_cuda = cfg_train["use_cuda"]
    num_epochs = cfg_train["num_epochs"]
    save_epochs = cfg_train["save_epochs"]
    
    # last layer activation
    ll_activation_name = cfg_train["ll_activation"]
    ll_activation = None
    if ll_activation_name == "softmax":
        ll_activation = nn.Softmax(0)
    elif ll_activation_name == "sigmoid":
        ll_activation = nn.Sigmoid()
    
    # loss function
    loss_name = cfg_train["loss"]
    loss = None
    if loss_name == "CE":
        loss = nn.CrossEntropyLoss(reduction="mean")
    elif loss_name == "BCE":
        loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    selection_criteria = "min_loss"
    batch_size = cfg_train["batch_size"]
    num_batches = len(se_train) // batch_size
    learning_rate = cfg_train["learning_rate"]
    weight_decay = cfg_train["weight_decay"]
    optimizer = cfg_train["optimizer"]
    num_clips = cfg_train["num_clips"]
    classes_names = cfg_train["classes_names"]
    classes_names_abb = cfg_train["classes_names_abb"]
    structured_events = cfg_train["structured_events"]
    
    saved_models_dir = cfg_dataset.saved_models_dir
    # signature
    train_info = "/{}/ne_{}_bs_{}_lr_{}_wd_{}_opt_{}/".format(run_id, num_epochs, batch_size, learning_rate,
                                                              weight_decay, optimizer)
    saved_models_dir += train_info
    os.makedirs(saved_models_dir, exist_ok=True)
    
    epochs_predictions = {
        "train":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [], "ground_truth": [],
                "outputs_act": [], "predictions": []
            },
        "val":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [], "ground_truth": [],
                "outputs_act": [], "predictions": []
            },
        "test":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [], "ground_truth": [],
                "outputs_act": [], "predictions": []
            }
    }
    
    # to save metrics during training
    writer_train = SummaryWriter(cfg_dataset.tf_logs_dir + train_info + "train/")
    writer_val = SummaryWriter(cfg_dataset.tf_logs_dir + train_info + "val/")
    brief_summary = open("{}/brief_summary.txt".format(cfg_dataset.tf_logs_dir + train_info), "w")
    best_model_ep = 0
    
    if optimizer == "Adam":
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    features_train = convert_to_float_tensor(features_train)
    features_test = convert_to_float_tensor(features_test)
    
    labels_train = get_labels(se_train, cfg_train)
    labels_train_textual = get_textual_label_from_tensor(labels_train, classes_names_abb)
    labels_val = get_labels(se_val, cfg_train)
    labels_val_textual = get_textual_label_from_tensor(labels_val, classes_names_abb)
    labels_test = get_labels(se_test, cfg_train)
    labels_test_textual = get_textual_label_from_tensor(labels_test, classes_names_abb)
    
    max_fmap_score = 0.
    
    num_training_examples = len(se_train)

    # fmap_score = evaluate(
    #     -1, "Validation", se_val[:5], features_train, labels_val, labels_val_textual, nn_model, loss,
    #     ll_activation, selection_criteria, num_clips, mnz_models, structured_events, use_cuda, classes_names,
    #     classes_names_abb, writer_val, brief_summary, epochs_predictions["val"]
    # )
    
    optimizer.zero_grad()
    rng = random.Random(cfg_train["seed"])
    for epoch in range(1, num_epochs + 1):
        start_time_epoch = time.time()
        print("\n--- START EPOCH {}\n".format(epoch))
        nn_model.train()
        
        rng.shuffle(se_train)
        
        tot_time_mnz = 0.
        epoch_loss = 0.
        batch_loss = 0.
        
        for index, example_train in enumerate(se_train):
            print(example_train)
            
            # get video, duration, num_features, se name and interval where the se is happening
            video, se_name, duration, num_features, se_interval = \
                example_train[0], example_train[1], example_train[2], example_train[3], example_train[4]
            
            features_video = features_train[video]
            
            # get clip and its labels
            features_clip = features_video[se_interval[0]:se_interval[1] + 1]
            example_id = "{}-{}-{}".format(video, se_name, se_interval)
            labels_clip_textual = labels_train_textual[example_id]
            
            # get the output from the network
            out = nn_model(features_clip.unsqueeze(0))
            
            outputs = out['final_output'][0]
            # mnz
            outputs_act = ll_activation(outputs.transpose(0, 1))
            
            # minizinc part
            tot_time_example = 0
            
            mnz_problem, _ = build_problem_exp1(se_name, mnz_models[se_name], outputs_act)
            
            start_time = time.time()
            sol = pymzn.minizinc(mnz_problem, solver=pymzn.gurobi)
            end_time = time.time()
            tot_time_example += end_time - start_time
            
            tot_time_mnz += tot_time_example
            
            mnz_pred = torch.zeros(outputs.shape)
            
            fill_mnz_pred_exp1(mnz_pred, sol, se_name)
            
            print("--- call to mnz - time = {:.2f}\n".format(tot_time_example))

            # example_loss = loss(outputs, mnz_pred)
            example_loss = loss(outputs, torch.argmax(mnz_pred, 1))
            batch_loss += example_loss
            
            example_loss.backward()
            
            print("MNZ sol: {}".format(str(sol)))
            print("Ground Thruth: {}".format(labels_clip_textual))
            
            print(
                "\nEpoch {} - example {}/{} ---- loss = {:.4f}\n".format(epoch, index + 1, num_training_examples,
                                                                         example_loss))
            
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
            epochs_predictions["train"]["ground_truth"].append(labels_clip.cpu().detach().numpy())
            epochs_predictions["train"]["outputs_act"].append(outputs_act.cpu().detach().numpy())
            epochs_predictions["train"]["predictions"].append(mnz_pred.cpu().detach().numpy())
        
        end_time_epoch = time.time()
        print("--- END EPOCH {} -- LOSS {} -- TIME {:.2f}-- TIME MNZ {:.2f}\n".format(
            epoch, epoch_loss, end_time_epoch - start_time_epoch, tot_time_mnz))
        
        writer_train.add_scalar("Loss", epoch_loss / num_training_examples, epoch)
        
        fmap_score = evaluate(
            epoch, "Validation", se_val, features_train, labels_val, labels_val_textual, nn_model, loss,
            ll_activation, selection_criteria, num_clips, mnz_models, structured_events, use_cuda, classes_names,
            classes_names_abb, writer_val, brief_summary, epochs_predictions["val"]
        )
        
        if fmap_score > max_fmap_score:
            best_model_ep = epoch
            max_fmap_score = fmap_score
        
        state = {
            "epoch": epoch,
            "state_dict": nn_model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(state, saved_models_dir + "model_{}.pth".format(epoch))
    
    best_model_path = saved_models_dir + "model_{}.pth".format(best_model_ep)
    print("Loading model " + best_model_path)
    
    # load and evaluate best model on test set
    if use_cuda:
        state = torch.load(best_model_path)
    else:
        state = torch.load(best_model_path, map_location=torch.device('cpu'))
    
    nn_model.load_state_dict(state["state_dict"])
    
    fmap_score = evaluate(
        best_model_ep, "Test", se_test, features_test, labels_test, labels_test_textual, nn_model,
        loss, ll_activation, selection_criteria, num_clips, mnz_models, structured_events, use_cuda, classes_names,
        classes_names_abb, None, brief_summary, epochs_predictions["test"]
    )
    
    with open("{}/epochs_predictions.pickle".format(cfg_dataset.tf_logs_dir + train_info), "wb") as epp_file:
        pickle.dump(epochs_predictions, epp_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    brief_summary.close()
    print(fmap_score)
    print(best_model_ep)
