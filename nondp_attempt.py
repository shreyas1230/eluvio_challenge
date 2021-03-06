import pickle
import torch
from torch.autograd import Variable
import numpy as np
import os
import glob
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from IPython import embed
import argparse

from evaluate_sceneseg import *

parser = argparse.ArgumentParser(description='train or test')

parser.add_argument('--train', '-tr', action="store_true")
parser.add_argument('--num_epochs', '-ne', type=int, default=10)
parser.add_argument('--model_num', '-mn', type=int, default=0)  # which model to load when testing


args = parser.parse_args()

class MovieSet(Dataset):
    def __init__(self, filenames):
        self.files = filenames
    def __len__(self):
        return len(filenames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = pickle.load(open(filenames[idx], "rb"))
        inp = f["scene_transition_boundary_prediction"]
        gt = f["scene_transition_boundary_ground_truth"]
        # print(f["place"].shape)
        # print(f["cast"].shape)
        # print(f["action"].shape)
        # print(f["audio"].shape)
        # max_features = f["place"].shape[1]
        # zero_pad = lambda x: torch.cat((x, torch.zeros((x.shape[0], max_features - x.shape[1]))), dim=1)
        # features = torch.stack((f["place"], zero_pad(f["cast"]), zero_pad(f["action"]), zero_pad(f["audio"])))
        # corr = []
        # for i in range(features.shape[0]):
        #     k = np.linalg.norm(features[i], axis=1)
        #     k = torch.Tensor(k).view((k.shape[0], 1))
        #     features[i] = features[i]/k
        # features[torch.isnan(features)] = 0
        # corr = torch.max(torch.sum(features[:, :-1] * features[:, 1:], dim=2), dim=0)[0]
        # embed()
        # raise Exception

        # shots = torch.cat([f["place"], f["cast"], f["action"], f["audio"]], dim=1)
        # k = np.linalg.norm(shots, axis=1)
        # k = torch.Tensor(k).view((k.shape[0],1))
        # shots = shots/k
        # corr = torch.sum(shots[:-1] * shots[1:], dim=1)
        # inp = torch.stack((inp.float(), corr))
        # embed()
        # raise Exception
        return inp.reshape((len(inp), 1)).float(), gt.reshape((gt.shape[0], 1)).float()

        # return inp.permute(1, 0), gt.reshape((gt.shape[0], 1)).float()

filenames = glob.glob(os.path.join("data", "tt*.pkl"))
data = MovieSet(filenames)

batch_size=1
num_epochs = args.num_epochs
loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nlayers = 2
n_directions = 1
h_size = 1

if args.train:
    model = nn.LSTM(input_size=1, hidden_size=h_size, num_layers=nlayers)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    best = np.inf
    best_epoch = 0
    for epoch in range(num_epochs):
        loss = 0
        print("EPOCH {}".format(epoch+1))
        for idx, (inp, label) in enumerate(loader):
            inp, label = inp.permute(1, 0, 2).to(device), label.permute(1, 0, 2).to(device)
            h0 = torch.randn(nlayers*n_directions, inp.shape[1], h_size).to(device)
            c0 = torch.randn(nlayers*n_directions, inp.shape[1], h_size).to(device)
            output, (hn, cn) = model(inp, (h0, c0))
            # params = []
            # for p in model.parameters():
            #     params.append(torch.clone(p.data).detach())
            # optimizer.zero_grad()
            loss += criterion(torch.sigmoid(output), label)
            # loss.backward()
            # optimizer.step()
            # for i, param in enumerate(model.parameters()):
            #     print(torch.sum((params[i] - param)**2))
            if idx % 30 == 0:
                print("batch {}".format(idx))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print("LOSS {}\n".format(loss.item()))
        if loss.item() < best:
            torch.save(model.state_dict(), "models/model_epoch_{}.pt".format(epoch))
            best = loss.item()
            best_epoch = epoch

    torch.save(model.state_dict(), "models/final_model.pt")
    print(best, best_epoch)

else:
    loaded_model = nn.LSTM(input_size=1, hidden_size=h_size, num_layers=nlayers)
    loaded_model.load_state_dict(torch.load("models/model_epoch_{}.pt".format(args.model_num)))
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    gt_dict = dict()
    pr_dict = dict()
    shot_to_end_frame_dict = dict()

    for fn in filenames:
        f = pickle.load(open(fn, "rb"))
        ground = f["scene_transition_boundary_ground_truth"]
        gt_dict[f["imdb_id"]] = ground

        inp = f["scene_transition_boundary_prediction"]
        # shots = torch.cat([x["place"], x["cast"], x["action"], x["audio"]], dim=1)
        # k = np.linalg.norm(shots, axis=1)
        # k = torch.Tensor(k).view((k.shape[0],1))
        # shots = shots/k
        # corr = torch.sum(shots[:-1] * shots[1:], dim=1)
        # inp = torch.stack((inp.float(), corr))

        # max_features = f["place"].shape[1]
        # zero_pad = lambda x: torch.cat((x, torch.zeros((x.shape[0], max_features - x.shape[1]))), dim=1)
        # features = torch.stack((f["place"], zero_pad(f["cast"]), zero_pad(f["action"]), zero_pad(f["audio"])))
        # corr = []
        # for i in range(features.shape[0]):
        #     k = np.linalg.norm(features[i], axis=1)
        #     k = torch.Tensor(k).view((k.shape[0], 1))
        #     features[i] = features[i]/k
        # features[torch.isnan(features)] = 0
        # corr = torch.max(torch.sum(features[:, :-1] * features[:, 1:], dim=2), dim=0)[0]
        # inp = torch.stack((inp.float(), corr))

        # inp = inp.reshape((inp.shape[1], 1, inp.shape[0])).to(device)
        inp = inp.reshape((len(inp), 1, 1)).float().to(device)
        h0 = torch.randn(nlayers*n_directions, inp.shape[1], h_size).to(device)
        c0 = torch.randn(nlayers*n_directions, inp.shape[1], h_size).to(device)
        out, (hn, cn) = loaded_model(inp, (h0, c0))
        out = torch.sigmoid(out).reshape(ground.shape)
        pr_dict[f["imdb_id"]] = out.detach().cpu().numpy()

        # pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]

        shot_to_end_frame_dict[f["imdb_id"]] = f["shot_end_frame"]


    scores = dict()

    scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
    print(scores["AP"], scores["mAP"])
