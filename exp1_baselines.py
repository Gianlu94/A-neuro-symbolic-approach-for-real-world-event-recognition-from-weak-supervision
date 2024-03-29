import os
import copy
import pickle
import random
import time

import numpy as np
import torch
from torch.autograd.variable import Variable
from torch import nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tensorboardX import SummaryWriter

from dataset import get_labels, get_avg_labels, get_se_labels
from utils import convert_to_float_tensor

        
def get_se_prediction_min_loss(outputs, avg_labels_clip, loss):
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
        epoch, mode, se_list, features, labels, new_labels, is_nn_for_ev,  nn_model, loss, ll_activation, num_clips, se_labels,
        use_cuda, classes_names, writer, brief_summary, epochs_predictions
):
    nn_model.eval()
    num_examples = len(se_list)
    se_names = list(se_labels.keys())
    num_se = len(se_names)
    
    actions_predictions = []
    actions_ground_truth = []
    
    #tot_loss = 0.
    print("\nStarting evaluation")
    start_time_ev = time.time()
    
    for i, example in enumerate(se_list):
        # gt_se (ground truth structured event)
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
        
        if is_nn_for_ev == 0:
            labels_clip = labels[example_id]
        elif is_nn_for_ev == 1 or is_nn_for_ev == 2:
            labels_clip = new_labels[example_id][gt_se_name]
            
        new_labels_clip = new_labels[example_id]
        
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
            outputs = out['final_output']
            outputs = outputs.squeeze(0)
            outputs = outputs[new_begin_se:new_end_se + 1]

            
            predicted_se_name = get_se_prediction_min_loss(outputs, new_labels_clip, loss)
            labels_clip_predicted_se = new_labels_clip[predicted_se_name]
            
            #example_loss = loss(outputs, avg_labels_clip_pred_se)
            #tot_loss += example_loss
            
            outputs_act = ll_activation(outputs)

            labels_clip = labels_clip.cpu().data.numpy()
            labels_clip_true_se = new_labels_clip[gt_se_name].cpu().data.numpy()
            labels_clip_predicted_se = labels_clip_predicted_se.cpu().data.numpy()
            
            assert len(outputs) == len(labels_clip_predicted_se)
            
            epochs_predictions["epoch"].append(epoch)
            epochs_predictions["video"].append(video)
            epochs_predictions["gt_se_names"].append(gt_se_name)
            epochs_predictions["pred_se_names"].append(predicted_se_name)
            epochs_predictions["se_interval"].append(se_interval)
            if mode == "Test":
                epochs_predictions["ground_truth"].append(labels_clip)
            else:
                epochs_predictions["ground_truth"].append(labels_clip_true_se)
                
            epochs_predictions["ground_truth_avg_pred_se"].append(labels_clip_predicted_se)
            epochs_predictions["outputs_act"].append(outputs_act.cpu().detach().numpy())
            
            if is_nn_for_ev == 0:
                new_outputs = set_outputs_for_metrics_computation(predicted_se_name, outputs_act)
                epochs_predictions["predictions"].append(new_outputs.cpu().detach().numpy())
                se_predictions = np.zeros((new_outputs.shape[0], num_se))
                se_predictions[:, se_labels[predicted_se_name]] = 1
                se_gt = np.zeros((new_outputs.shape[0], num_se))
                se_gt[:, se_labels[gt_se_name]] = 1

                
                if mode == "Test":
                    actions_ground_truth.extend(np.concatenate((labels_clip, se_gt), axis=1))
                else:
                    #avg labels
                    actions_ground_truth.extend(np.concatenate((labels_clip_true_se, se_gt), axis=1))
                
                actions_predictions.extend(np.concatenate((new_outputs, se_predictions), axis=1))
            elif is_nn_for_ev == 1:
                actions_ground_truth.extend(labels_clip_true_se)
                actions_predictions.extend(labels_clip_predicted_se)
            elif is_nn_for_ev == 2:
                # number of ae
                num_ae = len(classes_names) - len(se_labels)
                new_outputs = set_outputs_for_metrics_computation(predicted_se_name, outputs_act)
                new_outputs = np.concatenate((new_outputs[:, :num_ae], labels_clip[:, num_ae:]), 1)

                if mode == "Test":
                    actions_ground_truth.extend(labels_clip)
                else:
                    # avg labels
                    actions_ground_truth.extend(labels_clip_true_se)
                
                actions_predictions.extend(new_outputs)
                epochs_predictions["predictions"].append(new_outputs)
            

    actions_ground_truth = np.array(actions_ground_truth)
    actions_predictions = np.array(actions_predictions)
    
    # compute metrics
    actions_avg_precision_score = average_precision_score(actions_ground_truth, actions_predictions, average=None)
    
    actions_results = precision_recall_fscore_support(actions_ground_truth, actions_predictions, average=None)
    
    actions_f1_scores, actions_precision, actions_recall = actions_results[2], actions_results[0], actions_results[1]
    
    end_time_ev = time.time()
    
    metrics_to_print = """
        \nTIME: {:.2f}
        {} -- Epoch: {}, Precision per class: {}
        {} -- Epoch: {}, Recall per class: {}
        {} -- Epoch: {}, F1-Score per class: {}
        {} -- Epoch: {}, Average Precision: {}
        {} -- Epoch: {}, F1-Score: {:.4f}, mAP: {:.4f}
    """.format(
        end_time_ev - start_time_ev,
        mode, epoch, actions_precision,
        mode, epoch, actions_recall,
        mode, epoch, str(actions_f1_scores),
        mode, epoch, str(actions_avg_precision_score),
        mode, epoch, np.nanmean(actions_f1_scores), np.nanmean(actions_avg_precision_score)
    )

    print(metrics_to_print, flush=True)
    brief_summary.write(metrics_to_print)
    
    classes_to_metrics = copy.deepcopy(classes_names)
    if is_nn_for_ev == 0:
        classes_to_metrics += se_names
    
    if writer is not None:
        for i, class_name in enumerate(classes_to_metrics):
            # writer.add_scalar("Loss {} ".format(class_name), tot_loss.item()/num_se, epoch)
            writer.add_scalar("F1 Score {} ".format(class_name), actions_f1_scores[i], epoch)
            writer.add_scalar("Precision {} ".format(class_name), actions_precision[i], epoch)
            writer.add_scalar("Recall {} ".format(class_name), actions_recall[i], epoch)
            # writer.add_scalar("AP {} ".format(class_name), actions_avg_precision_score[i], epoch)
        
        writer.add_scalar('Avg F1 Score', np.nanmean(actions_f1_scores), epoch)
        writer.add_scalar('Avg Precision', np.nanmean(actions_precision), epoch)
        writer.add_scalar('Avg Recall', np.nanmean(actions_recall), epoch)
        writer.add_scalar('Avg AP', np.nanmean(actions_avg_precision_score), epoch)
    
    return np.nanmean(actions_avg_precision_score)


def train_exp1_neural(se_train, se_val, se_test, features_train, features_test, nn_model, cfg_train, cfg_dataset):
    
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
    elif ll_activation_name == "sigmoid":
        ll_activation = nn.Sigmoid()
        
    # loss function
    loss_name = cfg_train["loss"]
    loss = None
    if loss_name == "CE":
        loss = nn.CrossEntropyLoss(reduction="mean")
    elif loss_name == "BCE":
        loss = nn.BCEWithLogitsLoss()

    is_nn_for_ev = cfg_train["is_nn_for_ev"]
    batch_size = cfg_train["batch_size"]
    num_batches = len(se_train) // batch_size
    learning_rate = cfg_train["learning_rate"]
    weight_decay = cfg_train["weight_decay"]
    optimizer = cfg_train["optimizer"]
    num_clips = cfg_train["num_clips"]
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
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [],
                "ground_truth": [], "ground_truth_avg_pred_se": [], "outputs_act": [],
                "predictions": []
            },
        "val":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [],
                "ground_truth": [], "ground_truth_avg_pred_se": [], "outputs_act": [], "predictions": []
            },
        "test":
            {
                "epoch": [], "video": [], "gt_se_names": [], "pred_se_names": [], "se_interval": [],
                "ground_truth": [], "ground_truth_avg_pred_se": [], "outputs_act": [], "predictions": []
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

    labels_val = None
    labels_test = None
    # new_labels refer to avg labels or se_labels depending on the exp
    new_labels_train = None
    new_labels_val = None
    new_labels_test = None

    if is_nn_for_ev == 0 or is_nn_for_ev == 2: # 0 = nn for atomic events, 2 = nn for atomic and structured events
        labels_val = get_labels(se_val, cfg_train)
        labels_test = get_labels(se_test, cfg_train)
        new_labels_train = get_avg_labels(se_train, cfg_train)
        new_labels_val = get_avg_labels(se_val, cfg_train)
        new_labels_test = get_avg_labels(se_test, cfg_train)
        classes_names = cfg_train["classes_names"]
    elif is_nn_for_ev == 1: # 1 = nn for structured
        new_labels_train = get_se_labels(se_train, cfg_train)
        new_labels_val = get_se_labels(se_val, cfg_train)
        new_labels_test = get_se_labels(se_test, cfg_train)
        classes_names = list(structured_events.keys())
    
    max_fmap_score = 0.
    
    num_training_examples = len(se_train)
    
    optimizer.zero_grad()

    rng = random.Random(cfg_train["seed"])
    # fmap_score = evaluate(
    #     -1, "Validation", se_val, features_train, labels_val, new_labels_val, is_nn_for_ev, nn_model, loss,
    #     ll_activation,
    #     num_clips, structured_events, use_cuda, classes_names, writer_val, brief_summary,
    #     epochs_predictions["val"]
    # )
   
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
            
            true_labels_clip = new_labels_train[id_label][gt_se_name]

            # get the output from the network
            out = nn_model(features_clip.unsqueeze(0))
            
            outputs = out['final_output'][0]
            
            example_loss = loss(outputs, true_labels_clip)

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

            outputs_act = ll_activation(outputs)

            predicted_se_name = get_se_prediction_min_loss(outputs, new_labels_train[id_label], loss)
          
            labels_clip_pred_se = new_labels_train[id_label][predicted_se_name]
            
            epochs_predictions["train"]["epoch"].append(epoch)
            epochs_predictions["train"]["video"].append(video)
            epochs_predictions["train"]["gt_se_names"].append(gt_se_name)
            epochs_predictions["train"]["pred_se_names"].append(predicted_se_name)
            epochs_predictions["train"]["se_interval"].append(se_interval)
            epochs_predictions["train"]["ground_truth"].append(true_labels_clip.cpu().detach().numpy())
            epochs_predictions["train"]["ground_truth_avg_pred_se"].append(
                labels_clip_pred_se.cpu().detach().numpy())
            
            if is_nn_for_ev == 0:
                new_outputs = set_outputs_for_metrics_computation(predicted_se_name, outputs_act)
                epochs_predictions["train"]["predictions"].append(new_outputs.cpu().detach().numpy())
            elif is_nn_for_ev == 1:
                epochs_predictions["train"]["predictions"].append(labels_clip_pred_se.cpu().detach().numpy())
            elif is_nn_for_ev == 2:
                #filter ae based on se predicitons
                num_ae = cfg_train["classes"] - len(structured_events)
                new_outputs = set_outputs_for_metrics_computation(predicted_se_name, outputs_act)
                new_outputs = torch.cat((new_outputs[:, :num_ae], labels_clip_pred_se[:, num_ae:]), 1)
                epochs_predictions["train"]["predictions"].append(new_outputs.cpu().detach().numpy())

            epochs_predictions["train"]["outputs_act"].append(outputs_act.cpu().detach().numpy())
            
        end_time_epoch = time.time()
        print("--- END EPOCH {} -- LOSS {} -- TIME {:.2f}\n".format(epoch, epoch_loss, end_time_epoch - start_time_epoch))
        
        writer_train.add_scalar("Loss", epoch_loss / num_training_examples, epoch)
        
        fmap_score = evaluate(
            epoch, "Validation", se_val, features_train, labels_val, new_labels_val, is_nn_for_ev, nn_model, loss, ll_activation,
            num_clips, structured_events, use_cuda, classes_names, writer_val, brief_summary,
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
        best_model_ep, "Test", se_test, features_test, labels_test, new_labels_test, is_nn_for_ev, nn_model, loss, ll_activation, num_clips,
        structured_events, use_cuda, classes_names, None, brief_summary, epochs_predictions["test"]
    )

    with open("{}/epochs_predictions.pickle".format(cfg_dataset.tf_logs_dir + train_info), "wb") as epp_file:
        pickle.dump(epochs_predictions, epp_file, protocol=pickle.HIGHEST_PROTOCOL)

    brief_summary.close()
    print(fmap_score)
    print(best_model_ep)


def evaluate_test_set_with_proportion_rule_exp1(nn_model, se_test, features_test, cfg_train, cfg_dataset):
    exp_info = "/{}-{}/".format(cfg_train["run_id"], cfg_train["exp"])
    logs_dir = cfg_dataset.tf_logs_dir + exp_info
    os.makedirs(logs_dir, exist_ok=True)
    brief_summary = open("{}/brief_summary.txt".format(logs_dir), "w")

    is_nn_for_ev = cfg_train["is_nn_for_ev"]
    
    nn_model.eval()
    structured_events = cfg_train["structured_events"]
    num_examples = len(se_test)
    se_names = list(structured_events.keys())
    num_se = len(se_names)

    features_test = convert_to_float_tensor(features_test)
    if is_nn_for_ev == 1:
        cfg_train["classes"] = 12
    labels_test = get_labels(se_test, cfg_train)
    
    new_labels_test = None
    if is_nn_for_ev == 0:
        new_labels_test = get_avg_labels(se_test, cfg_train)
    elif is_nn_for_ev == 1:
        cfg_train["classes"] = len(structured_events)
        avg_labels_test = get_avg_labels(se_test, cfg_train)
        new_labels_test = get_se_labels(se_test, cfg_train)
        
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
    
    num_clips = cfg_train["num_clips"]
    use_cuda = cfg_train["use_cuda"]
    
    test_predictions = {
        "video": [], "se_interval": [], "gt_se_names": [], "pred_se_names": [], "ground_truth": [], "outputs_act": [], "predictions": []}
    actions_predictions = []
    actions_ground_truth = []
    
    #tot_loss = 0.
    print("\nStarting evaluation")
    start_time_ev = time.time()
    
    for i, example in enumerate(se_test):
        
        video, gt_se_name, duration, num_features, se_interval, _ = example
        
        print(
            "\nProcessing example [{}, {}, {}]  {}/{}  ".format(video, gt_se_name, (se_interval), i + 1, num_examples),
            end="")
        
        new_begin_se = 0
        new_end_se = se_interval[1] - se_interval[0]

        # get features for the current video
        features_video = np.array(features_test[video])
        features_video = Variable(torch.from_numpy(features_video).type(torch.FloatTensor))
        
        example_id = "{}-{}-{}".format(video, gt_se_name, se_interval)
        
        # get clip and its labels
        features_clip = features_video[se_interval[0]:se_interval[1] + 1]
        
        labels_clip = labels_test[example_id]
        
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
            outputs = out['final_output']
            outputs = outputs.squeeze(0)
            outputs = outputs[new_begin_se:new_end_se + 1]
            
            #example_loss = loss(outputs, labels_clip)  # labels_clip)
            #tot_loss += example_loss
            predicted_se_name = get_se_prediction_min_loss(outputs, new_labels_test[example_id], loss)
            
            outputs_act = ll_activation(outputs)
            labels_clip = labels_clip.cpu().data.numpy()
            
            assert len(outputs) == len(labels_clip)
            
            test_predictions["video"].append(video)
            test_predictions["se_interval"].append(se_interval)
            test_predictions["gt_se_names"].append(gt_se_name)
            test_predictions["pred_se_names"].append(predicted_se_name)
            test_predictions["ground_truth"].append(labels_clip)
            test_predictions["outputs_act"].append(outputs_act.data.cpu().detach().numpy())
            
            if is_nn_for_ev == 0:
                # avg labels are named as new_labels in this case
                new_outputs = new_labels_test[example_id][predicted_se_name].data.cpu().detach().numpy()
            elif is_nn_for_ev == 1:
                new_outputs = avg_labels_test[example_id][predicted_se_name].data.cpu().detach().numpy()
                
            test_predictions["predictions"].append(new_outputs)

            se_predictions = np.zeros((new_outputs.shape[0], num_se))
            se_predictions[:, structured_events[predicted_se_name]] = 1
            se_gt = np.zeros((new_outputs.shape[0], num_se))
            se_gt[:, structured_events[gt_se_name]] = 1

            # actions_predictions.extend(se_predictions)
            # actions_ground_truth.extend(se_gt)
            actions_predictions.extend(np.concatenate((new_outputs, se_predictions), axis=1))
            actions_ground_truth.extend(np.concatenate((labels_clip, se_gt), axis=1))
            
    actions_ground_truth = np.array(actions_ground_truth)
    actions_predictions = np.array(actions_predictions)
    
    # compute metrics
    actions_avg_precision_score = average_precision_score(actions_ground_truth, actions_predictions, average=None)
    
    actions_results = precision_recall_fscore_support(actions_ground_truth, actions_predictions, average=None)
    
    actions_f1_scores, actions_precision, actions_recall = actions_results[2], actions_results[0], actions_results[1]
    
    end_time_ev = time.time()
    
    metrics_to_print = """
                        \nTIME: {:.2f}
                        Test -- Precision per class: {}
                        Test -- Recall per class: {}
                        Test -- F1-Score per class: {}
                        Test -- Average Precision: {}
                        Test -- F1-Score: {:.4f}, mAP: {:.4f}
                    """.format(
        end_time_ev - start_time_ev,
        actions_precision,
        actions_recall,
        str(actions_f1_scores),
        str(actions_avg_precision_score),
        np.nanmean(actions_f1_scores), np.nanmean(actions_avg_precision_score)
    )
    
    print(metrics_to_print, flush=True)
    brief_summary.write(metrics_to_print)
    brief_summary.close()
    
    with open("{}/test_predictions.pickle".format(logs_dir), "wb") as tp_file:
        pickle.dump(test_predictions, tp_file, protocol=pickle.HIGHEST_PROTOCOL)

