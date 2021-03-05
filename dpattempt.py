import pickle
import torch
from torch.autograd import Variable
import numpy as np
import os
import glob
import time
import IPython
from IPython import embed

from evaluate_sceneseg import *

def findSuperShots(weights, shots, preds, threshold=0):
    """
    Inputs:
        weights - (N x 1) vector of weights for each shot
        shots - (N x D) matrix of shot representations
        preds - (N-1) length vector of predictions for each shot s whether shot boundary s:s+1 is a scene boundary

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

def g(corr):
    """
    Inputs:
        corr - (S x S) matrix of supershot correlations, S <= N

    Outputs:
        score - scalar optimal scene cut score achieved by sshot_set
    """
    if len(corr) == 0:
        return 0
    if len(corr) == 1:
        return 0.5
    N = corr.shape[0]
    corr_offd = corr.masked_select(~torch.eye(N, dtype=bool)).view(N, N-1).T
    Fs_total = torch.sum(corr_offd)/corr_offd.shape[0]
    Ft_total = torch.sum(torch.sigmoid(torch.max(corr_offd, dim=0)[0]))
    score = (Fs_total + Ft_total)/N
    return score


def get_g_mat(C):
    """
    Inputs:
        C - (C x D) matrix of supershot representations

    Outputs:
        g_mat - (C x C) matrix of optimal scene cut score for every possible consecutive supershot groupings
    """
    arr = torch.clone(C)
    k = torch.linalg.norm(arr, dim=1)
    k = k.view((k.shape[0],1))
    arr = arr/k
    corr = arr @ arr.T # correlation matrix

    g_mat = torch.zeros((C.shape[0], C.shape[0]))
    for col in range(C.shape[0]):
        for row in range(C.shape[0]):
            start = col
            end = row+1
            if end <= start:
                continue
            idx = torch.arange(start, end)
            val = g(corr[idx][:,idx])
            g_mat[row, col] = val
    return g_mat

def dp_sol(shots, initial_preds):
    """
    Inputs:
        shots - (N x D) matrix of shot representations
        initial_preds - (N-1) length vector of initial scene boundary predictions

    Outputs:
        new_preds - (N-1) length vector of scene boundary predictions
    """
    W = Variable(torch.ones((shots.shape[0], 1)), requires_grad=True)
    num_iterations = 6
    predictions = initial_preds
    for it in range(num_iterations):
        print("Iteration: {}".format(it))
        C, pred_bounds = findSuperShots(W, shots, predictions, threshold=0.7)
        print(len(C))
        G_mat = get_g_mat(C)
        print(G_mat)
        F = torch.zeros((C.shape[0], C.shape[0]-1, 2))
        F[:, 0, 0] = G_mat[:,0]

        for j in range(1, F.shape[1]):
            for i in range(j, F.shape[0]):
                F[i, j, 0] = torch.max(F[:i, j-1, 0] + G_mat[i, 1:i+1])
                F[i, j, 1] = torch.argmax(F[:i, j-1, 0] + G_mat[i, 1:i+1])

        ind = torch.argmax(F[-1,:,0])
        lst_idxs = [F[-1, ind, 1].long().detach().item()]
        for k in range(ind, -1, -1):
            lst_idxs = [int(F[lst_idxs[0], k, 1].long().detach().item())] + lst_idxs
        lst_idxs = torch.unique(torch.Tensor(lst_idxs), sorted=True)
        # print(len(lst_idxs))

        # embed()
        # raise Exception

        pred_idxs = []
        prev = -1

        # for i in lst_idxs:
        #     if i != prev and i < len(pred_bounds):
        #         pred_idxs.append(int(i))
        new_pred_idxs = np.array(pred_bounds)[lst_idxs.long().tolist()]

        predictions = np.zeros(len(predictions))
        # new_preds = np.array(predictions)
        predictions[new_pred_idxs] = 1

        X = W.detach().clone()
        value = torch.max(F[-1,:,0])
        value.backward()
        # print(torch.sum(W.grad.data)/W.grad.shape[0])
        # raise Exception
        W.data = W.data - 10 * W.grad.data
        W.grad.data.zero_()
        # print("Amount changed: {}".format(torch.sum((X-W)**2)))
        # raise Exception

    # value = F[F.shape[0]-1, F.shape[1]-1, 0]
    # value.backward()
    # W.data = W.data - 0.005 * W.grad.data
    # W.grad.data.zero_()
    initial_preds[predictions==1] = 1
    new_preds = initial_preds
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
    # C = findSuperShots(Variable(torch.ones((D.shape[0], 1)), requires_grad=True), D, predictions, threshold=0.9)
    # embed()
    # raise Exception

    final_preds = dp_sol(D, predictions)
    pr_dict[x["imdb_id"]] = torch.Tensor(final_preds)
    # break


scores = dict()

scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
print(scores["AP"], scores["mAP"])
