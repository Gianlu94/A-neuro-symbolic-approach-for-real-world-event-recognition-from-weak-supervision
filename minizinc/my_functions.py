import torch


def _create_atomic_actions_matrix(name, atomic_actions):
    aa_shape = atomic_actions.shape
    mnz_array = "array [1..{}, 1..{}] of int: atomic_actions;\n atomic_actions = [".format(aa_shape[0], aa_shape[1])
    
    content = ""
    # for each aa
    for i in range(aa_shape[0]):
        content += "| "
        # for each t
        for t in range(aa_shape[1]):
            content += " {},".format(round(atomic_actions[i][t].item()*1000))
        content += "\n"
    content = content[:-2]
    content += "|];\n"
    
    return mnz_array + content


def build_problem(se_name, model, nn_output, classes):
    se_begin = 0
    se_end = nn_output.shape[1] - 1
    
    data = ""
    atomic_actions = None

    if se_name == "high_jump":
        # index start from 1 in array (no zero)
        data += "bHJ = {}; \n eHJ = {};\n".format(se_begin+1, se_end+1)
        atomic_actions = torch.stack((nn_output[classes["Run"] - 1], nn_output[classes["Jump"] - 1], nn_output[classes["Fall"] - 1]),
                                     0)
    elif se_name == "long_jump":
        # index start from 1 in array (no zero)
        data += "bLJ = {}; \n eLJ = {};\n".format(se_begin + 1, se_end + 1)
        atomic_actions = torch.stack((nn_output[classes["Run"] - 1], nn_output[classes["Jump"] - 1], nn_output[classes["Sit"] - 1]),
                                     0)
    data += _create_atomic_actions_matrix(se_name, atomic_actions)

    return model + data, atomic_actions
    