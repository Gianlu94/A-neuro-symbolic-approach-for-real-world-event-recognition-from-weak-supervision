import argparse
import pickle

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd


# def create_matrix_to_plot(ground_truth, nn_pred, mnz_pred, combined, class_to_evaluate, colors):
#     num_rows = ground_truth.shape[0]
#     num_columns = ground_truth.shape[1]
#     matrix_to_plot = np.zeros((num_rows*4, num_columns))
#
#     labels = []
#     labels_colors = []
#     for i in range(0, num_rows*4, 4):
#         idx_action = i // 4
#         matrix_to_plot[i:i+4] = np.stack([
#             ground_truth[idx_action], (nn_pred[idx_action] > 0.5).astype(int), mnz_pred[idx_action],
#             (combined[idx_action] > 0.5).astype(int)], axis=0)
#
#         labels += [class_to_evaluate[idx_action] for _ in range(4)]
#         labels_colors += colors
#
#     return matrix_to_plot, labels, labels_colors

def create_matrix_to_plot(ground_truth, predictions, class_names, colors):
    num_rows, num_columns = ground_truth.shape
    
    matrix_to_plot = np.zeros((num_rows*2, num_columns))

    labels = []
    labels_colors = []
    
    for i in range(num_rows):
        matrix_to_plot[i*2:(i+1)*2] = np.stack([ground_truth[i], predictions[i]])
        labels += [class_names[i]] * 2
        labels_colors += colors

    return matrix_to_plot, labels, labels_colors


def _get_color(color):
    c = None
    if color == "g":
        c = [0.,   1.,   0.]
    elif color == "r":
        c = [1., 0., 0.]
    return c


def highlight_intervals(matrix, colors):
    highlighted_matrix = np.zeros((matrix.shape + (3,)))
    
    num_rows, num_columns = matrix.shape
    
    for i in range(num_rows):
        for j in range(num_columns):
            if matrix[i, j] == 0.:
                highlighted_matrix[i, j, :] = [1., 1., 1.]
            else:
                if (i + 1) % 2 != 0:
                    idx_color = 0
                else:
                    idx_color = 1
                    
                c = _get_color(colors[idx_color])
                highlighted_matrix[i, j, :] = c
    return highlighted_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot ground thruth and predictions")
    parser.add_argument("-pickle_file", type=str, help="path to pickle file of results")
    parser.add_argument("-path_to_plots", type=str, help="path where to save plots")
    parser.add_argument("-mode", type=str, help="Mode (train - val - test)")
    parser.add_argument("-exp", type=str, help="mnz or neural")

    args = parser.parse_args()
    mode = args.mode
    path_to_plots = args.path_to_plots + "/{}/".format(mode)
    exp = args.exp
    os.makedirs(path_to_plots, exist_ok=True)

    class_names = ["Run", "Jump", "Fall", "HT_WindUp", "HT_Spin", "HT_Release"]

    with open(args.pickle_file, "rb") as pf:
        epochs_predictions = pickle.load(pf)

    colors = ["g", "r"]
    legend_list = None

    if exp == "mnz":
        legend_list = ["ground_truth", "mnz"]
    elif exp == "neural":
        legend_list = ["ground_truth", "neural"]

    fig = plt.figure(figsize=(20, 10))

    epochs_predictions_df = pd.DataFrame(epochs_predictions[mode])
    epochs_predictions_grouped = epochs_predictions_df.groupby(["video", "se_interval"])
    for _, example_epochs in epochs_predictions_grouped:
        for example_epoch in example_epochs.iterrows():
            example_epoch = example_epoch[1]
            epoch, video, gt_se_name, pred_se_name, se_interval, ground_truth, predictions = example_epoch["epoch"],\
                example_epoch["video"], example_epoch["gt_se_names"], example_epoch["pred_se_names"], example_epoch["se_interval"],\
                example_epoch["ground_truth"], example_epoch["predictions"]

            ground_truth_tr = ground_truth.transpose()
            predictions_tr = predictions.transpose()
            #breakpoint()
            if pred_se_name == "HighJump":
                predictions_tr[3:, :] = 0
                unified_matrix, labels, colors_labels = create_matrix_to_plot(ground_truth_tr, predictions_tr, class_names, colors)
            elif pred_se_name == "HammerThrow":
                predictions_tr[:3, :] = 0
                unified_matrix, labels, colors_labels = create_matrix_to_plot(ground_truth_tr, predictions_tr, class_names, colors)

            highlighted_matrix = highlight_intervals(unified_matrix, colors)

            plt.title("epoch {}: {} - gt_se {} - pred_se {} - {}".format(epoch, video, gt_se_name, pred_se_name, se_interval))
            plt.yticks(range(0, len(labels)), labels)
            plt.xlim(0, predictions.shape[0])
            plt.xticks(range(0, predictions.shape[0]+1))
            [ytick.set_color(colors_labels[i]) for i, ytick in enumerate(plt.gca().get_yticklabels())]
            patches = [mpatches.Patch(color=colors[i], label="{}".format(legend_list[i])) for i in range(2)]
            plt.legend(handles=patches)
            plt.imshow(highlighted_matrix, aspect="auto")

            if mode == "test":
                path_to_example = "{}/{}-{}-se_interval_{}.png".format(path_to_plots, video, gt_se_name, se_interval)
                plt.savefig("{}".format(path_to_example))
            else:
                path_to_example = "{}{}-{}-se_interval_{}/".format(path_to_plots, video, gt_se_name, se_interval)
                os.makedirs(path_to_example, exist_ok=True)
                plt.savefig("{}epoch_{}.png".format(path_to_example, epoch))

            plt.cla()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Plot ground thruth and predictions")
#     parser.add_argument("-pickle_file", type=str, help="path to pickle file of results")
#     parser.add_argument("-path_to_plots", type=str, help="path where to save plots")
#     parser.add_argument("-mode", type=str, help="Mode (train - val - test)")
#     parser.add_argument("-exp", type=str, help="mnz or neural")
#
#     args = parser.parse_args()
#     mode = args.mode
#     path_to_plots = args.path_to_plots + "/{}/".format(mode)
#     exp = args.exp
#     os.makedirs(path_to_plots, exist_ok=True)
#
#     class_names = ["Run", "Jump", "Fall", "HT_WindUp", "HT_Spin", "HT_Release"]
#
#     with open(args.pickle_file, "rb") as pf:
#         epochs_predictions = pickle.load(pf)
#
#     colors = ["g", "r"]
#     legend_list = None
#
#     legend_list = ["ground_truth", "ground_truth_avg"]
#
#     fig = plt.figure(figsize=(20, 10))
#
#     epochs_predictions_df = pd.DataFrame(epochs_predictions[mode])
#     epochs_predictions_grouped = epochs_predictions_df.groupby(["video", "se_interval"])
#     for _, example_epochs in epochs_predictions_grouped:
#         for example_epoch in example_epochs.iterrows():
#             example_epoch = example_epoch[1]
#             epoch, video, gt_se_name, pred_se_name, se_interval, ground_truth, ground_truth_avg, predictions = example_epoch["epoch"],\
#                 example_epoch["video"], example_epoch["gt_se_names"], example_epoch["pred_se_names"], example_epoch["se_interval"],\
#                 example_epoch["ground_truth"], example_epoch["ground_truth_avg"], example_epoch["predictions"]
#
#             ground_truth_tr = ground_truth.transpose()
#             ground_truth_avg_tr = ground_truth_avg.transpose()
#
#             unified_matrix, labels, colors_labels = create_matrix_to_plot(ground_truth_tr, ground_truth_avg_tr, class_names, colors)
#
#             highlighted_matrix = highlight_intervals(unified_matrix, colors)
#
#             plt.title("epoch {}: {} - gt_se {} - pred_se {} - {}".format(epoch, video, gt_se_name, pred_se_name, se_interval))
#             plt.yticks(range(0, len(labels)), labels)
#             plt.xlim(0, predictions.shape[0])
#             plt.xticks(range(0, predictions.shape[0]+1))
#             [ytick.set_color(colors_labels[i]) for i, ytick in enumerate(plt.gca().get_yticklabels())]
#             patches = [mpatches.Patch(color=colors[i], label="{}".format(legend_list[i])) for i in range(2)]
#             plt.legend(handles=patches)
#             plt.imshow(highlighted_matrix, aspect="auto")
#
#
#             path_to_example = "{}/{}-{}-se_interval_{}.png".format(path_to_plots, video, gt_se_name, se_interval)
#             plt.savefig("{}".format(path_to_example))
#
#             plt.cla()
#             break
    