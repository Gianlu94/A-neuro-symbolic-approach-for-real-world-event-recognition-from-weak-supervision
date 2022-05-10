import os
import pickle
import random
import time

import numpy as np
import torch
from torch.autograd.variable import Variable
from torch import nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tensorboardX import SummaryWriter

from dataset import get_examples_direct_supervision, get_labels, get_se_labels
from minizinc.my_functions import build_problem_exp1, fill_mnz_pred_exp1
import pymzn
from pymzn import Status
from utils import convert_to_float_tensor


def get_se_prediction(outputs, avg_labels_clip, loss):
    se_labels = list(avg_labels_clip.keys())
    num_se = len(se_labels)
    scores = torch.zeros((num_se))
    
    for idx, current_se in enumerate(se_labels):
        scores[idx] = loss(outputs, avg_labels_clip[current_se])
    
    return se_labels[int(torch.argmin(scores))]


def set_outputs_for_metrics_computation(pred_se_name, outputs):
    tmp_outputs = outputs
    new_outputs = torch.zeros_like(tmp_outputs)
    
    if pred_se_name == "HighJump":
        tmp_outputs[:, 3:] = 0
    elif pred_se_name == "LongJump":
        tmp_outputs[:, 2] = 0
        tmp_outputs[:, 4:] = 0
    elif pred_se_name == "PoleVault":
        tmp_outputs[:, 3] = 0
        tmp_outputs[:, 5:] = 0
    elif pred_se_name == "HammerThrow":
        tmp_outputs[:, :5] = 0
        tmp_outputs[:, 8:] = 0
    elif pred_se_name == "ThrowDiscus":
        tmp_outputs[:, :8] = 0
        tmp_outputs[:, 10:] = 0
    elif pred_se_name == "Shotput":
        tmp_outputs[:, :10] = 0
    elif pred_se_name == "JavelinThrow":
        tmp_outputs[:, 1:11] = 0

    new_outputs[range(new_outputs.shape[0]), torch.argmax(tmp_outputs, 1)] = 1

    return new_outputs

    
def evaluate(
        epoch, mode, se_list, features, labels_ae, labels_se, nn_model, loss, ll_activation, ae_se_corr, num_clips,
        se_labels, use_cuda, classes_names, writer, brief_summary, epochs_predictions
):
    nn_model.eval()
    num_examples = len(se_list)
    se_names = list(se_labels.keys())
    num_se = len(se_names)
    
    events_tmp_predictions = []
    events_tmp_ground_truth = []

    se_predictions = []
    se_gt = []
    
    #tot_loss = 0.
    print("\nStarting evaluation")
    start_time_ev = time.time()
    
    for i, example in enumerate(se_list):
        
        video, gt_se_name, duration, num_features, se_interval, _ = example
        
        print("\nProcessing example [{}, {}, {}]  {}/{}  ".format(video, gt_se_name, (se_interval), i + 1, num_examples),
              end="")

        new_begin_se = 0
        new_end_se = se_interval[1] - se_interval[0]
        
        # get features for the current video
        features_video = np.array(features[video])
        features_video = Variable(torch.from_numpy(features_video).type(torch.FloatTensor))
        
        example_id = "{}-{}-{}".format(video, gt_se_name, se_interval)
        
        # get clip and its labels
        features_clip = features_video[se_interval[0]:se_interval[1] + 1]
        
        labels_ae_clip = labels_ae[example_id]
        labels_se_clip = labels_se[example_id]
        
        # labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
        with torch.no_grad():
            if num_clips > 0:
                if len(features_clip) < num_clips:
                    # padding
                    features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
                    features_clip = torch.cat((features_clip, features_to_append), 0)
                assert len(features_clip) > 0
            else:
                features_clip = torch.unsqueeze(features_clip, 0)
                labels_clip = torch.unsqueeze(labels_clip, 0)
            
            if use_cuda:
                features_clip = features_clip.cuda()
                labels_clip = labels_clip.cuda()
            
            # get the output from the network
            out = nn_model(features_clip)
            
            outputs_ae = out['final_output'].squeeze(0)[:(new_end_se+1)]
            outputs_se = out["final_output_se"].squeeze(0)[:(new_end_se+1)]

            pred_se_name = get_se_prediction(outputs_se, labels_se_clip, loss)
            outputs_act_ae = ll_activation(outputs_ae)
            outputs_act_se = ll_activation(outputs_se)
            if not ae_se_corr:
                new_outputs_ae = set_outputs_for_metrics_computation("", outputs_act_ae)
            else:
                new_outputs_ae = set_outputs_for_metrics_computation(pred_se_name, outputs_act_ae)

            #example_loss = loss(outputs, avg_labels_clip_pred_se)
            #tot_loss += example_loss
            
            labels_ae_clip = labels_ae_clip.cpu().detach().data.numpy()
            outputs_ae = outputs_ae.cpu().detach().numpy()
            outputs_se = outputs_se.cpu().detach().numpy()
            outputs_act_ae = outputs_act_ae.cpu().detach().numpy()
            outputs_act_se = outputs_act_se.cpu().detach().numpy()
            
            epochs_predictions["epoch"].append(epoch)
            epochs_predictions["video"].append(video)
            epochs_predictions["gt_se_names"].append(gt_se_name)
            epochs_predictions["pred_se_names"].append(pred_se_name)
            epochs_predictions["se_interval"].append(se_interval)
            epochs_predictions["ground_truth"].append(labels_ae_clip)
            epochs_predictions["raw_outputs_ae"].append(outputs_ae)
            epochs_predictions["raw_outputs_se"].append(outputs_se)
            epochs_predictions["outputs_act_ae"].append(outputs_act_ae)
            epochs_predictions["outputs_act_se"].append(outputs_act_se)
            epochs_predictions["predictions"].append(new_outputs_ae.cpu().detach().data.numpy())
            
            se_tmp_predictions = np.zeros((new_outputs_ae.shape[0], num_se))
            
            se_tmp_predictions[:, se_labels[pred_se_name]] = 1
            se_tmp_gt = np.zeros((outputs_se.shape[0], num_se))
            se_tmp_gt[:, se_labels[gt_se_name]] = 1
            
            events_tmp_predictions.extend(np.concatenate((new_outputs_ae, se_tmp_predictions), axis=1))
            events_tmp_ground_truth.extend(np.concatenate((labels_ae_clip, se_tmp_gt), axis=1))
            se_predictions.append(pred_se_name)
            se_gt.append(gt_se_name)
    
    events_tmp_ground_truth = np.array(events_tmp_ground_truth)
    events_tmp_predictions = np.array(events_tmp_predictions)

    se_gt = np.array(se_gt)
    se_predictions = np.array(se_predictions)
    
    # compute metrics
    events_tmp_avg_precision_score = average_precision_score(events_tmp_ground_truth, events_tmp_predictions, average=None)
    events_tmp_results = precision_recall_fscore_support(events_tmp_ground_truth, events_tmp_predictions, average=None)
    events_tmp_f1_scores, events_tmp_precision, events_tmp_recall = events_tmp_results[2], events_tmp_results[0], events_tmp_results[1]

    #se_avg_precision_score = average_precision_score(se_gt, se_predictions, average=None)
    se_results = precision_recall_fscore_support(se_gt, se_predictions, average=None)
    se_f1_scores, se_precision, se_recall = se_results[2], se_results[0], se_results[1]
    
    end_time_ev = time.time()
    
    metrics_to_print = """
        \nTIME: {:.2f}
    {} -- Epoch: {}, Tmp Precision per class: {}
    {} -- Epoch: {}, Tmp Recall per class: {}
    {} -- Epoch: {}, Tmp F1-Score per class: {}
    {} -- Epoch: {}, Tmp Average Precision: {}
    {} -- Epoch: {}, Tmp F1-Score: {:.4f}, Tmp mAP: {:.4f}
    """.format(
        end_time_ev - start_time_ev,
        #mode, epoch, tot_loss.item() / num_examples,
        mode, epoch, events_tmp_precision,
        mode, epoch, events_tmp_recall,
        mode, epoch, str(events_tmp_f1_scores),
        mode, epoch, str(events_tmp_avg_precision_score),
        mode, epoch, np.nanmean(events_tmp_f1_scores), np.nanmean(events_tmp_avg_precision_score)
    )
    
    metrics_to_print += """
    ---------------------------------------------------
        {} -- Epoch: {}, Precision per se class: {}
        {} -- Epoch: {}, Recall per se class: {}
        {} -- Epoch: {}, F1-Score per se class: {}
        {} -- Epoch: {}, F1-Score: {:.4f}
        """.format(
        # mode, epoch, tot_loss.item() / num_examples,
        mode, epoch, se_precision,
        mode, epoch, se_recall,
        mode, epoch, se_f1_scores,
        #mode, epoch, se_avg_precision_score,
        mode, epoch, np.nanmean(se_f1_scores),
    )
    
    print(metrics_to_print, flush=True)
    brief_summary.write(metrics_to_print)
    
    if writer is not None:
        for i, class_name in enumerate(classes_names + se_names):
            # writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("Tmp F1 Score {} ".format(class_name), events_tmp_f1_scores[i], epoch)
            writer.add_scalar("Tmp Precision {} ".format(class_name), events_tmp_precision[i], epoch)
            writer.add_scalar("Tmp Recall {} ".format(class_name), events_tmp_recall[i], epoch)
            # writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)
        
        #writer.add_scalar('Loss', tot_loss.item() / num_examples, epoch)
        writer.add_scalar('Tmp Avg F1 Score', np.nanmean(events_tmp_f1_scores), epoch)
        writer.add_scalar('Tmp Avg Precision', np.nanmean(events_tmp_precision), epoch)
        writer.add_scalar('Tmp Avg Recall', np.nanmean(events_tmp_recall), epoch)
        writer.add_scalar('Tmp Avg AP', np.nanmean(events_tmp_avg_precision_score), epoch)


    if writer is not None:
        for i, class_name in enumerate(se_names):
            # writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("Se -- F1 Score {} ".format(class_name), se_f1_scores[i], epoch)
            writer.add_scalar("Se -- Precision {} ".format(class_name), se_precision[i], epoch)
            writer.add_scalar("Se -- Recall {} ".format(class_name), se_recall[i], epoch)
            # writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)

        # writer.add_scalar('Loss', tot_loss.item() / num_examples, epoch)
        writer.add_scalar('Se F1 Score', np.nanmean(se_f1_scores), epoch)
        writer.add_scalar('Se Precision', np.nanmean(se_precision), epoch)
        writer.add_scalar('Se Recall', np.nanmean(se_recall), epoch)
        #writer.add_scalar('Se AP', np.nanmean(se_avg_precision_score), epoch)
    
    return np.nanmean(events_tmp_avg_precision_score)


def train_exp2_neural(se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset):
    
    # training info
    run_id = cfg_train["run_id"]
    use_cuda = cfg_train["use_cuda"]
    num_epochs = cfg_train["num_epochs"]
    save_epochs = cfg_train["save_epochs"]
    
    # last layer activation
    ll_activation_name = cfg_train["ll_activation"]
    ll_activation = None
    if ll_activation_name == "softmax":
        ll_activation = nn.Softmax(1)
        
    # loss function
    loss_name = cfg_train["loss"]
    loss = None
    if loss_name == "CE":
        loss = nn.CrossEntropyLoss(reduction="mean")
    
    ae_se_corr = cfg_train["ae_se_corr"]
    batch_size = cfg_train["batch_size"]
    num_batches = len(se_train) // batch_size
    learning_rate = cfg_train["learning_rate"]
    weight_decay = cfg_train["weight_decay"]
    optimizer = cfg_train["optimizer"]
    num_clips = cfg_train["num_clips"]
    classes_names = cfg_train["classes_names"]
    structured_events = cfg_train["structured_events"]
    se_direct_sup = cfg_train["structured_events_direct_sup"]
    
    saved_models_dir = cfg_dataset.saved_models_dir
    # signature
    train_info = "/{}/ne_{}_bs_{}_lr_{}_wd_{}_opt_{}/".format(run_id, num_epochs, batch_size, learning_rate,
                                                                weight_decay, optimizer)
    saved_models_dir += train_info
    os.makedirs(saved_models_dir, exist_ok=True)

    epochs_predictions = {
        "train":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [],
                "ground_truth": [], "raw_outputs_ae": [], "raw_outputs_se": [],
                "outputs_act_ae": [], "outputs_act_se": [], "predictions": []
            },
        "val":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [],
                "ground_truth": [], "raw_outputs_ae": [], "raw_outputs_se": [],
                "outputs_act_ae": [], "outputs_act_se": [], "predictions": []
            },
        "test":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [],
                "ground_truth": [], "raw_outputs_ae": [], "raw_outputs_se": [],
                "outputs_act_ae": [], "outputs_act_se": [], "predictions": []
            },
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

    # se_train = se_train[:5]
    # se_val = [se_val[1]] + [se_val[-1]]
    # se_test = [se_test[1]] + [se_test[-1]]
    examples_dir_sup = get_examples_direct_supervision(se_train, se_direct_sup, cfg_train["seed"])
    labels_ae_train = get_labels(se_train, cfg_train)
    labels_se_train = get_se_labels(se_train, cfg_train)

    labels_ae_val = get_labels(se_val, cfg_train)
    labels_se_val = get_se_labels(se_val, cfg_train)
    
    labels_ae_test = get_labels(se_test, cfg_train)
    labels_se_test = get_se_labels(se_test, cfg_train)
    
    max_fmap_score = 0.
    
    num_training_examples = len(se_train)
    
    optimizer.zero_grad()

    rng = random.Random(cfg_train["seed"])
    # fmap_score = evaluate_with_mnz(
    #     best_model_ep, "Test", se_test, features_test, labels_ae_test, labels_se_test, nn_model, loss, ll_activation,
    #     num_clips, structured_events, use_cuda, classes_names, None, brief_summary,
    #     epochs_predictions["test"], cfg_train["path_to_mnz_models"]
    # )
    # with open("{}/epochs_predictions.pickle".format(cfg_dataset.tf_logs_dir + train_info), "wb") as epp_file:
    #     pickle.dump(epochs_predictions, epp_file, protocol=pickle.HIGHEST_PROTOCOL)
    # breakpoint()
    
    for epoch in range(1, num_epochs + 1):
        start_time_epoch = time.time()
        print("\n--- START EPOCH {}\n".format(epoch))
        nn_model.train()
        rng.shuffle(se_train)
        
        epoch_loss = 0.
        batch_loss = 0.
        
        for index, example_train in enumerate(se_train):
            print(example_train)
            # get video, duration, num_features, se name and interval where the se is happening
            video, gt_se_name, duration, num_features, se_interval = \
                example_train[0], example_train[1], example_train[2], example_train[3], example_train[4]
            features_video = features_train[video]
            
            # get clip and its labels
            features_clip = features_video[se_interval[0]:se_interval[1] + 1]
            id_label = "{}-{}-{}".format(video, gt_se_name, se_interval)
            

            # get the output from the network
            out = nn_model(features_clip.unsqueeze(0))
            
            # labels for ae and se
            labels_clip_ae = None
            labels_clip_se = labels_se_train[id_label][gt_se_name]
            
            # network outputs for ae and se
            outputs_se = out['final_output_se'][0]
            outputs_ae = None
            
            # use dataset grount truth
            if example_train in examples_dir_sup:
                labels_clip_ae = labels_ae_train[id_label]
                outputs_ae = out['final_output'][0]
            
            if outputs_ae is None:
                # loss on structured events
                example_loss = loss(outputs_se, labels_clip_se) #labels_train[id_label])
            else:
                # sum of losses of atomic and strctured events
                example_loss = loss(outputs_se, labels_clip_se) + loss(outputs_ae, labels_clip_ae)
                
            batch_loss += example_loss
            
            example_loss.backward()
            
            print(
                "\nEpoch {} - example {}/{} ---- loss = {:.4f}\n".format(epoch, index + 1, num_training_examples, example_loss))
            
            # batch update
            if (index + 1) % batch_size == 0:
                print("\nEpoch {} BATCH ---- loss = {}\n".format(epoch, batch_loss))
                epoch_loss += batch_loss / num_batches
                batch_loss = 0.
                optimizer.step()
                optimizer.zero_grad()


            # outputs_act = ll_activation(outputs)
            epochs_predictions["train"]["epoch"].append(epoch)
            epochs_predictions["train"]["video"].append(video)
            epochs_predictions["train"]["gt_se_names"].append(gt_se_name)
            epochs_predictions["train"]["se_interval"].append(se_interval)
            epochs_predictions["train"]["ground_truth"].append(labels_ae_train[id_label].cpu().detach().numpy())
            if outputs_ae is not None:
                epochs_predictions["train"]["raw_outputs_ae"].append(outputs_ae.cpu().detach().numpy())
                epochs_predictions["train"]["outputs_act_ae"].append(ll_activation(outputs_ae).cpu().detach().numpy())
            else:
                epochs_predictions["train"]["raw_outputs_ae"].append(outputs_ae)
                epochs_predictions["train"]["outputs_act_ae"].append(outputs_ae)

            epochs_predictions["train"]["raw_outputs_se"].append(outputs_se.cpu().detach().numpy())
            epochs_predictions["train"]["outputs_act_se"].append(ll_activation(outputs_se).cpu().detach().numpy())
            
        end_time_epoch = time.time()
        print("--- END EPOCH {} -- LOSS {} -- TIME {:.2f}\n".format(epoch, epoch_loss, end_time_epoch - start_time_epoch))
        
        writer_train.add_scalar("Loss", epoch_loss / num_training_examples, epoch)
        
        fmap_score = evaluate(
            epoch, "Validation", se_val, features_train, labels_ae_val, labels_se_val, nn_model, loss, ll_activation,
            ae_se_corr, num_clips, structured_events, use_cuda, classes_names, writer_val, brief_summary,
            epochs_predictions["val"]
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
        best_model_ep, "Test", se_test, features_test, labels_ae_test, labels_se_test, nn_model, loss, ll_activation,
        ae_se_corr, num_clips, structured_events, use_cuda, classes_names, None, brief_summary, epochs_predictions["test"]
    )

    with open("{}/epochs_predictions.pickle".format(cfg_dataset.tf_logs_dir + train_info), "wb") as epp_file:
        pickle.dump(epochs_predictions, epp_file, protocol=pickle.HIGHEST_PROTOCOL)

    brief_summary.close()
    print(fmap_score)
    print(best_model_ep)


def evaluate_with_mnz(
        epoch, mode, se_list, features, labels_ae, labels_se, nn_model, loss, ll_activation, num_clips,
        se_labels, use_cuda, classes_names, writer, brief_summary, epochs_predictions, path_to_mnz_models
):
    
    nn_model.eval()
    num_examples = len(se_list)
    se_names = list(se_labels.keys())
    num_se = len(se_names)

    # minizinc models for each structured event (se)
    mnz_files_names = os.listdir(path_to_mnz_models)
    mnz_files_names.sort()
    mnz_models = {}
    for mnz_file_name in mnz_files_names:
        se_name = mnz_file_name.split(".")[0]
        with open(path_to_mnz_models + mnz_file_name, "r") as mnz_file:
            mnz_models[se_name] = mnz_file.read()

    events_tmp_predictions = []
    events_tmp_ground_truth = []
    
    se_predictions = []
    se_gt = []

    # tot_loss = 0.
    print("\nStarting evaluation")
    start_time_ev = time.time()
    
    for i, example in enumerate(se_list):
        
        video, gt_se_name, duration, num_features, se_interval, _ = example
        
        print(
            "\nProcessing example [{}, {}, {}]  {}/{}  ".format(video, gt_se_name, (se_interval), i + 1, num_examples),
            end="")
        
        new_begin_se = 0
        new_end_se = se_interval[1] - se_interval[0]
        
        # get features for the current video
        features_video = np.array(features[video])
        features_video = Variable(torch.from_numpy(features_video).type(torch.FloatTensor))
        
        example_id = "{}-{}-{}".format(video, gt_se_name, se_interval)
        
        # get clip and its labels
        features_clip = features_video[se_interval[0]:se_interval[1] + 1]
        
        labels_ae_clip = labels_ae[example_id]
        labels_se_clip = labels_se[example_id]
        
        # labels_clip = labels_video[interval_cut_f[0]:interval_cut_f[1] + 1]
        with torch.no_grad():
            if num_clips > 0:
                if len(features_clip) < num_clips:
                    # padding
                    features_to_append = torch.zeros(num_clips - len(features_clip) % num_clips, features_clip.shape[1])
                    features_clip = torch.cat((features_clip, features_to_append), 0)
                assert len(features_clip) > 0
            else:
                features_clip = torch.unsqueeze(features_clip, 0)
                labels_clip = torch.unsqueeze(labels_clip, 0)
            
            if use_cuda:
                features_clip = features_clip.cuda()
                labels_clip = labels_clip.cuda()
            
            # get the output from the network
            out = nn_model(features_clip)
            
            outputs_ae = out['final_output'].squeeze(0)[:(new_end_se + 1)]
            outputs_se = out["final_output_se"].squeeze(0)[:(new_end_se + 1)]
            
            pred_se_name = get_se_prediction(outputs_se, labels_se_clip, loss)

            outputs_act_ae = ll_activation(outputs_ae)
            outputs_act_se = ll_activation(outputs_se)

            mnz_problem, _ = build_problem_exp1(pred_se_name, mnz_models[pred_se_name], outputs_act_ae.transpose(0, 1))

            start_time = time.time()

            sol = pymzn.minizinc(mnz_problem, solver=pymzn.gurobi)

            mnz_pred_ae = torch.zeros(outputs_ae.shape)

            if sol.status == Status.UNSATISFIABLE:
                print("*********SKIPPED EXAMPLE - GT SE {} -- PRED SE {}".format(gt_se_name, pred_se_name))
            else:
                fill_mnz_pred_exp1(mnz_pred_ae, sol, pred_se_name)
                
            
            # example_loss = loss(outputs, avg_labels_clip_pred_se)
            # tot_loss += example_loss
            
            labels_ae_clip = labels_ae_clip.cpu().detach().data.numpy()
            outputs_ae = outputs_ae.cpu().detach().numpy()
            outputs_se = outputs_se.cpu().detach().numpy()
            outputs_act_se = outputs_act_se.cpu().detach().numpy()
            
            epochs_predictions["epoch"].append(epoch)
            epochs_predictions["video"].append(video)
            epochs_predictions["gt_se_names"].append(gt_se_name)
            epochs_predictions["pred_se_names"].append(pred_se_name)
            epochs_predictions["se_interval"].append(se_interval)
            epochs_predictions["ground_truth"].append(labels_ae_clip)
            epochs_predictions["raw_outputs_ae"].append(outputs_ae)
            epochs_predictions["raw_outputs_se"].append(outputs_se)
            epochs_predictions["outputs_act_se"].append(outputs_act_se)
            

            if not sol.status == Status.UNSATISFIABLE:
                epochs_predictions["predictions"].append(mnz_pred_ae.cpu().detach().data.numpy())
                se_tmp_predictions = np.zeros((mnz_pred_ae.shape[0], num_se))
                se_tmp_predictions[:, se_labels[pred_se_name]] = 1
                se_tmp_gt = np.zeros((mnz_pred_ae.shape[0], num_se))
                se_tmp_gt[:, se_labels[gt_se_name]] = 1
                
                events_tmp_predictions.extend(np.concatenate((mnz_pred_ae, se_tmp_predictions), axis=1))
                events_tmp_ground_truth.extend(np.concatenate((labels_ae_clip, se_tmp_gt), axis=1))

                if "predictions_from_nn" in epochs_predictions:
                    epochs_predictions["predictions_from_nn"].append(torch.argmax(outputs_act_ae, 1))
            else:
                epochs_predictions["predictions"].append("unsat")
                if "predictions_from_nn" in epochs_predictions:
                    epochs_predictions["predictions_from_nn"].append("unsat")
            
            outputs_act_ae = outputs_act_ae.cpu().detach().numpy()
            epochs_predictions["outputs_act_ae"].append(outputs_act_ae)
            
            se_predictions.append(pred_se_name)
            se_gt.append(gt_se_name)
    
    events_tmp_ground_truth = np.array(events_tmp_ground_truth)
    events_tmp_predictions = np.array(events_tmp_predictions)
    
    se_gt = np.array(se_gt)
    se_predictions = np.array(se_predictions)
    
    # compute metrics
    events_tmp_avg_precision_score = average_precision_score(events_tmp_ground_truth, events_tmp_predictions,
                                                             average=None)
    events_tmp_results = precision_recall_fscore_support(events_tmp_ground_truth, events_tmp_predictions, average=None)
    events_tmp_f1_scores, events_tmp_precision, events_tmp_recall = events_tmp_results[2], events_tmp_results[0], \
                                                                    events_tmp_results[1]

    # se_avg_precision_score = average_precision_score(se_gt, se_predictions, average=None)
    se_results = precision_recall_fscore_support(se_gt, se_predictions, average=None)
    se_f1_scores, se_precision, se_recall = se_results[2], se_results[0], se_results[1]
    
    end_time_ev = time.time()
    
    metrics_to_print = """
        \nTIME: {:.2f}
    {} -- Epoch: {}, Tmp Precision per class: {}
    {} -- Epoch: {}, Tmp Recall per class: {}
    {} -- Epoch: {}, Tmp F1-Score per class: {}
    {} -- Epoch: {}, Tmp Average Precision: {}
    {} -- Epoch: {}, Tmp F1-Score: {:.4f}, Tmp mAP: {:.4f}
    """.format(
        end_time_ev - start_time_ev,
        # mode, epoch, tot_loss.item() / num_examples,
        mode, epoch, events_tmp_precision,
        mode, epoch, events_tmp_recall,
        mode, epoch, str(events_tmp_f1_scores),
        mode, epoch, str(events_tmp_avg_precision_score),
        mode, epoch, np.nanmean(events_tmp_f1_scores), np.nanmean(events_tmp_avg_precision_score)
    )
    
    metrics_to_print += """
    ---------------------------------------------------
        {} -- Epoch: {}, Precision per se class: {}
        {} -- Epoch: {}, Recall per se class: {}
        {} -- Epoch: {}, F1-Score per se class: {}
        {} -- Epoch: {}, F1-Score: {:.4f}
        """.format(
        # mode, epoch, tot_loss.item() / num_examples,
        mode, epoch, se_precision,
        mode, epoch, se_recall,
        mode, epoch, se_f1_scores,
        # mode, epoch, se_avg_precision_score,
        mode, epoch, np.nanmean(se_f1_scores),
    )
    
    print(metrics_to_print, flush=True)
    brief_summary.write(metrics_to_print)
    
    if writer is not None:
        for i, class_name in enumerate(classes_names + se_names):
            # writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("Tmp F1 Score {} ".format(class_name), events_tmp_f1_scores[i], epoch)
            writer.add_scalar("Tmp Precision {} ".format(class_name), events_tmp_precision[i], epoch)
            writer.add_scalar("Tmp Recall {} ".format(class_name), events_tmp_recall[i], epoch)
            # writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)

        # writer.add_scalar('Loss', tot_loss.item() / num_examples, epoch)
        writer.add_scalar('Tmp Avg F1 Score', np.nanmean(events_tmp_f1_scores), epoch)
        writer.add_scalar('Tmp Avg Precision', np.nanmean(events_tmp_precision), epoch)
        writer.add_scalar('Tmp Avg Recall', np.nanmean(events_tmp_recall), epoch)
        writer.add_scalar('Tmp Avg AP', np.nanmean(events_tmp_avg_precision_score), epoch)
    
    if writer is not None:
        for i, class_name in enumerate(se_names):
            # writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("Se -- F1 Score {} ".format(class_name), se_f1_scores[i], epoch)
            writer.add_scalar("Se -- Precision {} ".format(class_name), se_precision[i], epoch)
            writer.add_scalar("Se -- Recall {} ".format(class_name), se_recall[i], epoch)
            # writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)
        
        # writer.add_scalar('Loss', tot_loss.item() / num_examples, epoch)
        writer.add_scalar('Se F1 Score', np.nanmean(se_f1_scores), epoch)
        writer.add_scalar('Se Precision', np.nanmean(se_precision), epoch)
        writer.add_scalar('Se Recall', np.nanmean(se_recall), epoch)
        # writer.add_scalar('Se AP', np.nanmean(se_avg_precision_score), epoch)
    
    return np.nanmean(events_tmp_avg_precision_score)