import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from .utils import (
    cur_time,
    get_device,
    init_weights,
    get_args_parser,
    eval_cross_entropy_loss,
    eval_ndcg_at_k
)
from .metrics import NDCG
from .data_loader.load_mslr import DataLoader

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG = {
    "num_features": 136
}

TRAIN_DATA_PATH = "ranking/data/mslr-web10k/Fold1/train.txt"
VALI_DATA_PATH = "ranking/data/mslr-web10k/Fold1/vali.txt"


class LambdaRank(nn.Module):
    def __init__(self, sigma):
        super(LambdaRank, self).__init__()
        # set up the network, activations
        self.fc1 = nn.Linear(CONFIG["num_features"], 16)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigma = sigma
        self.activation = nn.Sigmoid()

    def forward(self, x):
        input1 = self.act1(self.fc1(x))
        return self.activation(self.fc2(input1)) * self.sigma


def train(
        epoch=10,
        batch_size=200,
        lr=0.0001,
        optim="adam",
        ndcg_reduction_type="log2",
        sigma=1,
        output_dir="tmp/ranking_output/"
):
    precision = torch.float32
    writer = SummaryWriter(output_dir)
    # get training and validation data:
    train_loader = DataLoader(TRAIN_DATA_PATH)
    train_loader.process_raw_data()
    vali_loader = DataLoader(VALI_DATA_PATH)
    df_vali = vali_loader.process_raw_data()
    net = LambdaRank(sigma)
    device = get_device()
    net.to(device)
    net.apply(init_weights)
    logger.info(net)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError("Optimization method {} not implemented".format(optim))

    ideal_ndcg = NDCG(2**8, ndcg_reduction_type)
    for i in range(epoch):
        logger.info("start epoch {}".format(i))
        net.train()
        net.zero_grad()
        count = 0
        grad_batch, y_pred_batch = [], []

        for X, Y in train_loader.process_query_batch():
            if np.sum(Y) == 0:
                continue

            denom = 1.0 / ideal_ndcg.maxDCG(Y)
            x_tensor = torch.tensor(X, dtype=precision, device=device)
            y_pred = net(x_tensor)
            y_pred_batch.append(y_pred)
            score_diff = 1.0 + torch.exp(sigma * (y_pred - y_pred.t()))
            # compute rank
            rank_df = pd.DataFrame({"Y": Y, "ind": np.arange(Y.shape[0])})
            rank_df = rank_df.sort_values("Y").reset_index(drop=True)
            rank_order = rank_df.sort_values("ind").index.values + 1

            with torch.no_grad():
                y_tensor = torch.tensor(Y, dtype=precision, device=device).view(-1, 1)
                rel_diff = y_tensor - y_tensor.t()
                pos_pairs = (rel_diff > 0).type(precision)
                neg_pairs = (rel_diff < 0).type(precision)
                Sij = pos_pairs - neg_pairs
                if ndcg_reduction_type == "log2":
                    gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
                elif ndcg_reduction_type == "identity":
                    gain_diff = y_tensor - y_tensor.t()
                else:
                    raise ValueError("ndcg_reduction method not supported yet {}".format(ndcg_reduction_type))

                rank_order_tensor = torch.tensor(rank_order, dtype=precision, device=device).view(-1, 1)
                decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)
                delta_ndcg = torch.abs(denom * gain_diff * decay_diff)
                lambda_i = sigma * (0.5 * (1 - Sij) - 1 / score_diff) * delta_ndcg
                lambda_i = torch.sum(lambda_i, 1, keepdim=True)

                assert lambda_i.shape == y_tensor.shape
                check_grad = torch.sum(lambda_i, (0, 1)).item()
                if check_grad == float('inf') or np.isnan(check_grad):
                    import ipdb
                    ipdb.set_trace()
                grad_batch.append(lambda_i)

            count += 1
            if count % batch_size == 0:
                logger.info("{} query processed".format(count))
                for grad, y_pred in zip(grad_batch, y_pred_batch):
                    y_pred.backward(grad / batch_size)
                optimizer.step()
                net.zero_grad()
                grad_batch, y_pred_batch = [], []

            if count % (5 * batch_size) == 0:
                logger.info(cur_time() + " eval for count: {}".format(count))
                eval_cross_entropy_loss(net, device, vali_loader, i, writer)
                eval_ndcg_at_k(net, device, df_vali, vali_loader, 100000, [10, 30], i, writer)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train()
