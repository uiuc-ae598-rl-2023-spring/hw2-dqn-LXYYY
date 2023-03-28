import json
import re


def read_json_file(file_path):
    # Read JSON data from file
    with open(file_path) as f:
        json_data = f.read()

    return json.loads(json_data)


# example description: DQN_ht64t64t_l0.001_e1_ed0.99_em0.1
def get_parameters_from_description(description):
    # get parameters from description
    seq = description.split('_')
    # split by :
    parameters = {}
    for s in seq[1:]:
        kv = s.split(':')
        parameters[kv[0]] = kv[1]

    h = parameters['h']
    lr = float(parameters['lr'])
    e = float(parameters['e'])
    ed = float(parameters['ed']) if 'ed' in parameters else 1.0
    em = float(parameters['em']) if 'em' in parameters else 0.0
    return h, lr, e, ed, em
