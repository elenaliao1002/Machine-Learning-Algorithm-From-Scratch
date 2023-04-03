import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["user", "item"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, seed=23):
        super(MF, self).__init__()
        torch.manual_seed(seed)
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # init 
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)

    def forward(self, u, v):
        ### BEGIN SOLUTION
        u_emb = self.user_emb(u)
        v_emb = self.item_emb(v)
        u_bias = self.user_bias(u)
        v_bias = self.item_bias(v)
        #add sigma to the output
        return torch.sigmoid((u_emb * v_emb).sum(1) + u_bias.squeeze() + v_bias.squeeze())
        # return (u_emb * v_emb).sum(1) + u_bias.squeeze() + v_bias.squeeze()
        ### END SOLUTION
    
def train_one_epoch(model, train_df, optimizer):
    """ Trains the model for one epoch"""
    model.train()
    ### BEGIN SOLUTION
    u = torch.LongTensor(train_df.user.values)
    v = torch.LongTensor(train_df.item.values)
    y = torch.FloatTensor(train_df.rating.values)
    y_hat = model(u, v)
    BCE = nn.BCELoss()
    # binary cross entropy loss
    loss = BCE(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss = loss.item()    
    ### END SOLUTION
    return train_loss

def valid_metrics(model, valid_df):
    """Computes validation loss and accuracy"""
    model.eval()
    ### BEGIN SOLUTION
    u = torch.LongTensor(valid_df.user.values)
    v = torch.LongTensor(valid_df.item.values)
    y = torch.FloatTensor(valid_df.rating.values)
    y_hat = model(u, v)
    # binary cross entropy loss using BCEWithLogitsLoss
    # valid_loss = torch.nn.BCEWithLogitsLoss(y_hat, y)
    BCE = nn.BCELoss()
    valid_loss = BCE(y_hat, y).item()
    valid_acc = ((y_hat > 0.5) == y).count_nonzero().item() / len(y)   
    ### END SOLUTION
    return valid_loss, valid_acc


def training(model, train_df, valid_df, epochs=10, lr=0.01, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        train_loss = train_one_epoch(model, train_df, optimizer)
        valid_loss, valid_acc = valid_metrics(model, valid_df) 
        print("train loss %.3f valid loss %.3f valid acc %.3f" % (train_loss, valid_loss, valid_acc)) 

