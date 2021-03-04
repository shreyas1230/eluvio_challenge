import pickle
import torch
from torch.autograd import Variable
import numpy as np
import os
import glob
import time

from evaluate_sceneseg import *

def findSuperShots(weights, shots, preds, threshold=0):
    """
    Inputs:
        weights - (N x 1) vector of weights for each shot
        shots - (N x D) matrix of shot representations
        preds - length N-1 vector of predictions for each shot s whether shot boundary s:s+1 is a scene boundary

    Output:
        out - (C x D) matrix of supershot representations where C is number of predicted supershots
    """
    predicted_boundaries = set([i for i in range(len(preds)) if preds[i] > threshold])
    idxs = []
    count = 0
    for i in range(len(preds)+1):
        idxs.append(count)
        if i in predicted_boundaries:
            count+=1
    idxs = torch.Tensor(idxs).reshape((len(idxs),1)).repeat(1, shots.shape[1]).long()
    out = torch.zeros(count+1, idxs.shape[1], dtype=shots.dtype).scatter_add_(0, idxs, weights*shots)
    return out, [i for i in range(len(preds)) if preds[i] > threshold]

def g(sshot_set):
    """
    Inputs:
        sshot_set - (S x D) matrix of supershot representations

    Outputs:
        score - scalar optimal scene cut score achieved by sshot_set
    """
    if len(sshot_set) == 0:
        return 0
    elif len(sshot_set) == 1:
        return 0.5
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    arr = torch.clone(sshot_set)
    k = torch.linalg.norm(arr, dim=1)
    k = k.view((k.shape[0],1))
    arr = arr/k
    new_arr = arr @ arr.T # correlation matrix
    new_arr_offd = new_arr.masked_select(~torch.eye(sshot_set.shape[0], dtype=bool)).view(sshot_set.shape[0], sshot_set.shape[0]-1).T
    Fs_total = torch.sum(new_arr_offd)/new_arr_offd.shape[0]
    Ft_total = torch.sum(torch.sigmoid(torch.max(new_arr_offd, dim=0)[0]))
    score = (Fs_total + Ft_total)/sshot_set.shape[0]
    return score

def get_g_mat(C):
    g_mat = torch.zeros((C.shape[0], C.shape[0]))
    for col in range(C.shape[0]):
        for row in range(1, C.shape[0]):
            start = col+1
            end = col+row+1
            if end > C.shape[0]:
                break

            idx = torch.arange(start, end)
            g_mat[row, col] = g(torch.index_select(C, 0, idx))
    return g_mat

def dp_sol(shots, initial_preds):
    W = Variable(torch.ones((shots.shape[0], 1)), requires_grad=True)
    C, pred_bounds = findSuperShots(W, shots, initial_preds, threshold=0.9)

    G_mat = get_g_mat(C)

    F = torch.zeros((C.shape[0], C.shape[0]-1, 2))
    for row in range(C.shape[0]):
        idx = torch.arange(row+1)
        F[row,0,0] = g(torch.index_select(C, 0, idx))

    for j in range(1, F.shape[1]):
        for i in range(F.shape[0]):

            F[i, j, 0] = torch.max(F[:i+1, j-1, 0] + G_mat[torch.flip(torch.arange(i+1),[0]), torch.arange(i+1)])
            F[i, j, 1] = torch.argmax(F[:i+1, j-1, 0] + G_mat[torch.flip(torch.arange(i+1),[0]), torch.arange(i+1)])

    lst_idxs = [F.shape[0]-1]
    for k in range(F.shape[1]-1, -1, -1):
        lst_idxs = [int(F[lst_idxs[0], k, 1].item())] + lst_idxs

    lst_idxs = torch.unique(torch.Tensor(lst_idxs), sorted=True)

    pred_idxs = []
    prev = -1

    for i in lst_idxs:
        if i != prev and i < len(pred_bounds):
            pred_idxs.append(int(i))
    new_pred_idxs = np.array(pred_bounds)[pred_idxs]

    new_preds = np.zeros(len(initial_preds))
    new_preds[new_pred_idxs] = 1

    # value = F[F.shape[0]-1, F.shape[1]-1, 0]
    # value.backward()
    # W.data = W.data - 0.005 * W.grad.data
    # W.grad.data.zero_()
    return new_preds


filenames = glob.glob(os.path.join("data", "tt*.pkl"))

gt_dict = dict()
pr_dict = dict()
shot_to_end_frame_dict = dict()

for fn in filenames:
    x = pickle.load(open(fn, "rb"))
    gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"]
    pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]
    shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]
    lst = [i for i in range(len(x["scene_transition_boundary_ground_truth"])) if x["scene_transition_boundary_ground_truth"][i] == True]

    predictions = x["scene_transition_boundary_prediction"].float()
    D = torch.cat([x["place"], x["cast"], x["action"], x["audio"]], dim=1)
    C = findSuperShots(Variable(torch.ones((D.shape[0], 1)), requires_grad=True), D, predictions, threshold=0.9)


    final_preds = dp_sol(D, predictions)
    pr_dict[x["imdb_id"]] = torch.Tensor(final_preds)


scores = dict()

scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
print(scores["AP"], scores["mAP"])
