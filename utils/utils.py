import os
import errno
import torch
from torch.autograd import Variable
import copy
import numpy as np
import glob
import string
import hashlib
import pickle


# Generalized from:
#https://stackoverflow.com/questions/18683821/generating-random-correlated-x-and-y-points-using-numpy
def gen_gaussian_data(means, covs, num):
    vals = np.random.multivariate_normal(means, covs, num).T
    for i, v in enumerate(vals):
        vals[i] = [int(x) for x in v]
    return list(zip(*vals))

def save_object(file_name, data):
    with open(file_name, "wb") as f:
        res = f.write(pickle.dumps(data))

def load_object(file_name):
    res = None
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            res = pickle.loads(f.read())
    return res

def save_or_update(obj_name, obj):
    # TODO: generalize this functionality
    dir_name = os.path.dirname(obj_name)
    if not os.path.exists(dir_name):
        make_dir(dir_name)
    saved_obj = load_object(obj_name)
    if saved_obj is None:
        saved_obj = obj
    else:
        if isinstance(saved_obj, dict):
            saved_obj.update(obj)
        elif isinstance(saved_obj, list):
            saved_obj.append(obj)
        else:
            # TODO: not sure best way to handle pandas
            saved_obj = saved_obj.append(obj)
    save_object(obj_name, saved_obj)

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

def cosine_similarity_vec(vec1, vec2):
    cosine_similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*
            np.linalg.norm(vec2))
    return cosine_similarity

def get_substr_words(words, substr):
    vals = []
    for w in words:
        if substr in w:
            vals.append(w)
    return vals

def get_regex_match_words(words, regex):
    vals = []
    for w in words:
        if regex.search(w) is not None:
            vals.append(w)
    return vals

def clear_terminal_output():
    os.system('clear')

def to_variable(arr, use_cuda=True):
    if isinstance(arr, list) or isinstance(arr, tuple):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        arr = Variable(torch.from_numpy(arr), requires_grad=True)
    else:
        arr = Variable(arr, requires_grad=True)

    if torch.cuda.is_available() and use_cuda:
        arr = arr.cuda()
    return arr


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def copy_network(Q):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2

def save_network(model, name, step, out_dir, delete_old=False):
    '''
    saves the model for the given step, and deletes models for older
    steps.
    '''
    out_dir = '{}/models/'.format(out_dir)
    # Make Dir
    make_dir(out_dir)
    # find files in the directory that match same format:
    fnames = glob.glob(out_dir + name + "*")
    if delete_old:
        for f in fnames:
            # delete old ones
            os.remove(f)

    # Save model
    torch.save(model.state_dict(), '{}/{}_step_{}'.format(out_dir, name, step))

def model_name_to_step(name):
    return int(name.split("_")[-1])

def get_model_names(name, out_dir):
    '''
    returns sorted list of the saved model_step files.
    '''
    out_dir = '{}/models/'.format(out_dir)
    # Make Dir
    # find files in the directory that match same format:
    fnames = sorted(glob.glob(out_dir + name + "*"), key=model_name_to_step)
    return fnames

def get_model_name(args):
    if args.suffix == "":
        return str(hash(str(args)))
    else:
        return args.suffix

def adjust_learning_rate(args, optimizer, epoch):
    """
    FIXME: think about what makes sense for us?
    Sets the learning rate to the initial LR decayed by half every 30 epochs
    """
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.5 ** (epoch // 30))
    lr = max(lr, args.min_lr)
    if (epoch % 30 == 0):
        print("new lr is: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
