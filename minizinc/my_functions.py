import pymzn
from pymzn import Status
import torch


def get_best_sol(sols, criteria, output, loss=None):
    scores = torch.zeros(len(sols))
    actions = []
    se_intervals = []

    for idx_sol, sol in enumerate(sols):
        if sol.status == Status.COMPLETE:
            action_names = list(sol[0].keys())

            if "bR" in action_names and "bPVPP" in action_names and "bJ" in action_names and "bF" in action_names:
                action = "PoleVault"
            elif "bR" in action_names and "bJ" in action_names and "bF" in action_names:
                action = "HighJump"
            elif "bR" in action_names and "bJ" in action_names and "bS" in action_names:
                action = "LongJump"
            elif "bHT_WU" in action_names and "bHT_S" in action_names and "bHT_R" in action_names:
                action = "HammerThrow"
            elif "bTD_WU" in action_names and "bTD_R" in action_names:
                action = "ThrowDiscus"
            elif "bSPB" in action_names and "bT" in action_names:
                action = "Shotput"
            elif "bR" in action_names and "bT" in action_names:
                action = "JavelinThrow"
                
            actions.append(action)
            
            mnz_pred = list(sol[0].values())
            mnz_pred = [t - 1 for t in mnz_pred]
            
            se_begin, se_end = mnz_pred[0], mnz_pred[-1]
            se_intervals.append([se_begin, se_end])
            
            class_of_interest = None
            rows = [list(range(mnz_pred[i], mnz_pred[i+1]+1)) for i in range(0, len(mnz_pred), 2)]
            columns = []
            if action == "HighJump": # [0, 1, 2]
                class_of_interest = [0, 1, 2]
            elif action == "LongJump": # [0, 1, 3]
                class_of_interest = [0, 1, 3]
            elif action == "PoleVault": # [0, 4, 1, 2]
                class_of_interest = [0, 4, 1, 2]
            elif action == "HammerThrow": # [5, 6, 7]
                class_of_interest = [5, 6, 7]
            elif action == "ThrowDiscus": # [8, 9]
                class_of_interest = [8, 9]
            elif action == "Shotput": # [10, 11]
                class_of_interest = [10, 11]
            elif action == "JavelinThrow": # [0, 11]
                class_of_interest = [0, 11]

            assert class_of_interest is not None
            for idx, c in enumerate(class_of_interest):
                columns.extend([c] * len(rows[idx]))

            rows = get_flatted_list(rows)
            #columns = get_flatted_list(columns)
            assert len(rows) == len(columns)
            
            if criteria == "max_avg":
                scores[idx_sol] = torch.mean(output[rows, columns])
            elif criteria == "min_loss":
                labels_from_mnz_pred = torch.zeros_like(output)
                labels_from_mnz_pred[rows, columns] = 1
                scores[idx_sol] = loss(output, labels_from_mnz_pred)
                
        elif sol.status == Status.UNSATISFIABLE:
            actions.append("Unsat")
            if criteria == "max_avg":
                scores[idx_sol] = 0.
            elif criteria == "min_loss":
                scores[idx_sol] = 10000.
            se_intervals.append([-1, -1])
    
    if criteria == "max_avg":
        idx_selected_action = torch.argmax(scores)
    elif criteria == "min_loss":
        idx_selected_action = torch.argmin(scores)
        
    return sols[idx_selected_action], actions[idx_selected_action], se_intervals[idx_selected_action]


def get_flatted_list(list):
    flatted_list = [item for sublist in list for item in sublist]
    return flatted_list


def fill_mnz_pred_exp1(mnz_pred, sol, se_name):
    if se_name == "HighJump":
        class_of_interest = [0, 1, 2]
    elif se_name == "LongJump":
        class_of_interest = [0, 1, 3]
    elif se_name == "PoleVault":
        class_of_interest = [0, 4, 1, 2]
    elif se_name == "HammerThrow":
        class_of_interest = [5, 6, 7]
    elif se_name == "ThrowDiscus":
        class_of_interest = [8, 9]
    elif se_name == "Shotput":
        class_of_interest = [10, 11]
    elif se_name == "JavelinThrow":
        class_of_interest = [0, 11]
    
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


def build_problem_exp1(se_name, model, nn_output, avg_actions_durations_in_f=None):
    # index start from 1 in array (no zero)
    se_begin = 1
    se_end = nn_output.shape[1]

    data = ""
    actions_predictions = None

    if se_name == "HighJump":
        data += "bHJ = {}; \n eHJ = {};\n".format(se_begin, se_end)
        #actions_predictions = torch.stack((nn_output[0], nn_output[1], nn_output[2], nn_output[3]), 0)
        actions_predictions = torch.stack((nn_output[0], nn_output[1], nn_output[2]), 0)
    elif se_name == "LongJump":
        data += "bLJ = {}; \n eLJ = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[0], nn_output[1], nn_output[3]), 0)
    elif se_name == "PoleVault":
        data += "bPV = {}; \n ePV = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[0], nn_output[4], nn_output[1], nn_output[2]), 0)
    elif se_name == "HammerThrow":
        data += "bHT = {}; \n eHT = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[5], nn_output[6], nn_output[7]), 0)
    elif se_name == "ThrowDiscus":
        data += "bTD = {}; \n eTD = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[8], nn_output[9]), 0)
    elif se_name == "Shotput":
        data += "bSP = {}; \n eSP = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[10], nn_output[11]), 0)
    elif se_name == "JavelinThrow":
        data += "bJT = {}; \n eJT = {};\n".format(se_begin, se_end)
        actions_predictions = torch.stack((nn_output[0], nn_output[11]), 0)
        
    data += _create_actions_matrix(actions_predictions)
    
    if avg_actions_durations_in_f is not None:
        # print("\n" + str(avg_actions_durations_in_f))
        for action, avg_duration in avg_actions_durations_in_f.items():
            data += "{} = {};\n".format(action, avg_duration)

    return model + data, actions_predictions


def set_prop_avg(se_name, se_avg_label, classe_names_abb):
    se_avg_label_tr = se_avg_label.transpose(0, 1)
    avg_actions_duration_f = {se_name: {}}
    if se_name == "HighJump":
        class_of_interest = [0, 1, 2]
    elif se_name == "LongJump":
        class_of_interest = [0, 1, 3]
    elif se_name == "PoleVault":
        class_of_interest = [0, 4, 1, 2]
    elif se_name == "HammerThrow":
        class_of_interest = [5, 6, 7]
    elif se_name == "ThrowDiscus":
        class_of_interest = [8, 9]
    elif se_name == "Shotput":
        class_of_interest = [10, 11]
    elif se_name == "JavelinThrow":
        class_of_interest = [0, 11]
        
    for c in class_of_interest:
        avg_duration_event = se_avg_label_tr[c].tolist().count(1.0)
        avg_actions_duration_f[se_name]["avg{}".format(classe_names_abb[c])] = avg_duration_event
    
    return avg_actions_duration_f
    
    