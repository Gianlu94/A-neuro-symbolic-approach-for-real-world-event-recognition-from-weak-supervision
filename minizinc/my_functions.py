import torch


def get_best_sol(sols, criteria, output, classes_names):
    scores = torch.zeros(len(sols))
    actions = []
    se_intervals = []
    for idx_sol, sol in enumerate(sols):
        action_names = list(sol[0].keys())
        if "bR" in action_names and "bJ" in action_names and "bF" in action_names:
            action = "HighJump"
        elif "bR" in action_names and "bJ" in action_names and "bS" in action_names:
            action = "LongJump"
        elif "bHT_WU" in action_names and "bHT_S" in action_names and "bHT_R" in action_names:
            action = "HammerThrow"
            
        actions.append(action)
        
        mnz_pred = list(sol[0].values())
        mnz_pred = [t - 1 for t in mnz_pred]
        
        se_begin, se_end = mnz_pred[0], mnz_pred[-1]
        se_intervals.append([se_begin, se_end])

        if criteria == "max_avg":
            rows = [list(range(mnz_pred[i], mnz_pred[i+1]+1)) for i in range(0, output.shape[1], 2)]
            if action == "HighJump":
                columns = [[i] * len(rows[i]) for i in range(3)]
            elif action == "HammerThrow":
                columns = [[i] * len(rows[i%3]) for i in range(3, 6)]
            rows = get_flatted_list(rows)
            columns = get_flatted_list(columns)
            scores[idx_sol] = torch.mean(output[rows, columns])
            
    idx_max_action = torch.argmax(scores)
    
    return sols[idx_max_action], actions[idx_max_action], se_intervals[idx_max_action]


def get_flatted_list(list):
    flatted_list = [item for sublist in list for item in sublist]
    return flatted_list


def fill_mnz_pred_exp1(mnz_pred, sol, se_name):
    if se_name == "HighJump":
        class_of_interest = [0, 1, 2]
    elif se_name == "HammerThrow":
        class_of_interest = [3, 4, 5]
    
    time_points = list(sol[0].values())
    # index start from 0
    time_points = [t - 1 for t in time_points]
    
    rows = [list(range(time_points[i], time_points[i + 1] + 1)) for i in range(0, len(time_points), 2)]
    columns = [len(r) * [class_of_interest[i]] for i, r in enumerate(rows)]
    
    rows = get_flatted_list(rows)
    columns = get_flatted_list(columns)
    
    assert len(rows) == len(columns)
    mnz_pred[rows, columns] = 1
    
    return (time_points[:2])

    
def _create_actions_matrix(actions_predictions):
    aa_shape = actions_predictions.shape
    #mnz_array = "array [1..{}, 1..{}] of float: actions_predictions;\n actions_predictions = [".format(aa_shape[0], aa_shape[1])
    mnz_array = "array [1..{}, 1..{}] of int: actions_predictions;\n actions_predictions = [".format(aa_shape[0],                                                                                                aa_shape[1])
    content = ""
    # for each aa
    for i in range(aa_shape[0]):
        content += "| "
        # for each t
        for t in range(aa_shape[1]):
            content += " {},".format(round(actions_predictions[i][t].item()*1000))
            #content += " {},".format(round(actions_predictions[i][t].item(), 3))
        content += "\n"
    content = content[:-2]
    content += "|];\n"
    return mnz_array + content


def build_problem_exp1(se_name, model, nn_output, avg_actions_durations_in_f):
    # index start from 1 in array (no zero)
    se_begin = 1
    se_end = nn_output.shape[1]

    data = ""
    actions_predictions = None

    if se_name == "HighJump":
        data += "bHJ = {}; \n eHJ = {};\n".format(se_begin, se_end)
        #actions_predictions = torch.stack((nn_output[0], nn_output[1], nn_output[2], nn_output[3]), 0)
        actions_predictions = torch.stack((nn_output[0], nn_output[1], nn_output[2]), 0)
    elif se_name == "HammerThrow":
        data += "bHT = {}; \n eHT = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[3], nn_output[4], nn_output[5]), 0)
        #actions_predictions = torch.stack((nn_output[4], nn_output[5], nn_output[6], nn_output[7]), 0)
        
    data += _create_actions_matrix(actions_predictions)
    
    if avg_actions_durations_in_f is not None:
        # print("\n" + str(avg_actions_durations_in_f))
        for action, avg_duration in avg_actions_durations_in_f.items():
            data += "{} = {};\n".format(action, avg_duration)

    return model + data, actions_predictions
    