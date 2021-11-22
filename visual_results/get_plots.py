import argparse
import pickle

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def create_matrix_to_plot(ground_truth, nn_pred, mnz_pred, combined, class_to_evaluate, colors):
    num_rows = ground_truth.shape[0]
    num_columns = ground_truth.shape[1]
    matrix_to_plot = np.zeros((num_rows*4, num_columns))
    
    labels = []
    labels_colors = []
    for i in range(0, num_rows*4, 4):
        idx_action = i // 4
        matrix_to_plot[i:i+4] = np.stack([
            ground_truth[idx_action], (nn_pred[idx_action] > 0.5).astype(int), mnz_pred[idx_action],
            (combined[idx_action] > 0.5).astype(int)], axis=0)
    
        labels += [class_to_evaluate[idx_action] for _ in range(4)]
        labels_colors += colors

    return matrix_to_plot, labels, labels_colors


def _get_color(color):
    c = None
    if color == "g":
        c = [0.,   1.,   0.]
    elif color == "b":
        c = [0., 0., 1.]
    elif color == "r":
        c = [1., 0., 0.]
    return c


def highlight_intervals(matrix, colors):
    highlighted_matrix = np.zeros((matrix.shape +(3,)))
   
    num_rows = matrix.shape[0]
    num_columns = matrix.shape[1]

    for i in range(num_rows):
        for j in range(num_columns):
            if matrix[i, j] == 0.:
                highlighted_matrix[i, j, :] = [1., 1., 1.]
            else:
                if (i + 1) % 4 == 0:
                    if matrix[i-1, j] == 1:
                        idx_color = (i - 1) % 4
                    else:
                        idx_color = (i - 2) % 4
                else:
                    idx_color = i % 4
                c = _get_color(colors[idx_color])
                highlighted_matrix[i, j, :] = c
    return highlighted_matrix

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get visual plot of the prediction")
    parser.add_argument("--pickle_file", type=str, help="path to pickle file of results")
    parser.add_argument("--path_to_plots", type=str, help="path where to save plots")

    args = parser.parse_args()
    path_to_plots = args.path_to_plots
    os.makedirs(path_to_plots, exist_ok=True)
    class_to_evaluate = ["Run", "Jump", "Fall", "Sit", "HighJump", "LongJump"]
    
    filtered_outputs = {}
    with open(args.pickle_file, "rb") as pf:
        filtered_outputs = pickle.load(pf)

    #plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 10))
    colors = ["g", "b", "r", "y"]
    legend_list = ["ground_truth", "nn", "nn+mnz"]
    for sample, values in filtered_outputs.items():
        ground_truth, ntw_pred, mnz_pred, combined = values["ground_truth"], values["n"], values["nmnz"], values["combined"]
        unified_matrix, labels, colors_labels = create_matrix_to_plot(ground_truth, ntw_pred, mnz_pred, combined, class_to_evaluate, colors)
        highlighted_matrix = highlight_intervals(unified_matrix, colors)
        
        plt.title(sample)
        plt.yticks(range(0, len(labels)), labels)
        [ytick.set_color(colors_labels[i]) for i, ytick in enumerate(plt.gca().get_yticklabels())]
        patches = [mpatches.Patch(color=colors[i], label="{}".format(legend_list[i])) for i in range(3)]
        plt.legend(handles=patches)
        plt.imshow(highlighted_matrix,  aspect="auto")

        plt.savefig("{}/{}.png".format(path_to_plots, sample))
    