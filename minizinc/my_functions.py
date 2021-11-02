import torch


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


def build_problem(se_name, model, nn_output, classes):
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

    return model + data, actions_predictions
    