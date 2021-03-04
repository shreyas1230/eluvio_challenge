import pickle
import torch
from torch.autograd import Variable
import numpy as np
import os
import glob
import time

from evaluate_sceneseg import *

# class Model(nn.Module)

filenames = glob.glob(os.path.join("data", "tt*.pkl"))

gt_dict = dict()
pr_dict = dict()
shot_to_end_frame_dict = dict()

for fn in filenames:
    x = pickle.load(open(fn, "rb"))
    gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"]
    pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]
    shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]
    # print(x["place"].shape)
    # print(x["cast"].shape)
    # print(x["action"].shape)
    # print(x["audio"].shape)
    # print(x["place"].shape)
    # lst = [i for i in range(len(x["scene_transition_boundary_ground_truth"])) if x["scene_transition_boundary_ground_truth"][i] == True]

    # predictions = x["scene_transition_boundary_prediction"].float()
    # D = torch.cat([x["place"], x["cast"], x["action"], x["audio"]], dim=1)
    # C = findSuperShots(Variable(torch.ones((D.shape[0], 1)), requires_grad=True), D, predictions, threshold=0.9)

    # print(C.shape)
    # print([i for i in range(len(predictions)) if predictions[i] > 0.9])
    # print(x["scene_transition_boundary_ground_truth"])
    # G_mat = get_g_mat(C)
    # print(g(C))
    # raise Exception


scores = dict()

scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
print(scores["AP"], scores["mAP"])
