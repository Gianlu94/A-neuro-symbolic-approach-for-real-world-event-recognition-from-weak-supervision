import torch


def get_best_sol(sols, criteria, output, dataset_classes):
    scores = torch.zeros(len(sols))
    actions = []
    se_intervals = []
    for idx_sol, sol in enumerate(sols):
        begin_name = list(sol[0].keys())[0]
        if "HJ" in begin_name:
            action = "HighJump"
        elif "LJ" in begin_name:
            action = "LongJump"
        actions.append(action)
        
        mnz_pred = list(sol[0].values())
        se_begin, se_end = mnz_pred[0]-1, mnz_pred[1]-1
        se_intervals.append([se_begin, se_end])
        if criteria == "max_avg":
            scores[idx_sol] = torch.mean(output[se_begin:se_end + 1, dataset_classes[action] - 1])
    
    idx_max_action = torch.argmax(scores)
    
    return sols[idx_max_action], actions[idx_max_action], se_intervals[idx_max_action]


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
    
    return class_of_interest[1:] + [class_of_interest[0]]

    
def _create_actions_matrix(actions_predictions):
    aa_shape = actions_predictions.shape
    mnz_array = "array [1..{}, 1..{}] of int: actions_predictions;\n actions_predictions = [".format(aa_shape[0], aa_shape[1])
    
    content = ""
    # for each aa
    for i in range(aa_shape[0]):
        content += "| "
        # for each t
        for t in range(aa_shape[1]):
            content += " {},".format(round(actions_predictions[i][t].item()*1000))
        content += "\n"
    content = content[:-2]
    content += "|];\n"
    
    return mnz_array + content


def build_problem(se_name,  model, nn_output, classes, avg_actions_durations_in_f):
    # index start from 1 in array (no zero)
    se_begin = 1
    se_end = nn_output.shape[1]
    
    data = ""
    actions_predictions = None

    if se_name == "high_jump":
        data += "bC = {}; \n eC = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[classes["Run"] - 1], nn_output[classes["Jump"] - 1],
                                          nn_output[classes["Fall"] - 1], nn_output[classes["HighJump"] - 1]), 0)
    elif se_name == "long_jump":
        data += "bC = {}; \n eC = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[classes["Run"] - 1], nn_output[classes["Jump"] - 1],
                                           nn_output[classes["Sit"] - 1], nn_output[classes["LongJump"] - 1]), 0)
    data += _create_actions_matrix(actions_predictions)

    if avg_actions_durations_in_f is not None:
        #print("\n" + str(avg_actions_durations_in_f))
        for action, avg_duration in avg_actions_durations_in_f.items():
            data += "{} = {};\n".format(action, avg_duration)

    return model + data, actions_predictions
    