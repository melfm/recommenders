#!/usr/bin/env python
import argparse
import sys
import os
import numpy as np
import pandas as pd 

from collections import defaultdict
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.datasets.amazon_reviews import get_review_data
from recommenders.datasets.split_utils import filter_k_core

# Transformer Based Models
from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.ssept import SSEPT

# Sampler for sequential prediction
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reviews_Books_5")
    parser.add_argument("--num_epochs", default=500)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--model", default="sasrec")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    # Set None for non-deterministic result
    RANDOM_SEED = 100

    data_dir = os.path.join("..", "tests", "resources", "deeprec", "sasrec")

    # Amazon Available Datasets
    # "reviews_Books_5", "reviews_Electronics_5"
    # "reviews_Beauty_5"
    dataset = args.dataset

    print('Training on ', dataset, '\n')
    print('Running model ', args.model)

    lr = 0.001             # learning rate
    maxlen = 50            # maximum sequence length for each user
    num_blocks = 2         # number of transformer blocks
    hidden_units = 100     # number of units in the attention calculation
    num_heads = 1          # number of attention heads
    dropout_rate = 0.1     # dropout rate
    l2_emb = 0.0           # L2 regularization coefficient
    num_neg_test = 100     # number of negative examples per positive example
    # 'sasrec' or 'ssept'
    model_name = args.model

    reviews_name = dataset + '.json'
    outfile = dataset + '.txt'

    reviews_file = os.path.join(data_dir, reviews_name)
    if not os.path.exists(reviews_file):
        reviews_output = get_review_data(reviews_file)
    else:
        reviews_output = os.path.join(data_dir, dataset+".json_output")

    if not os.path.exists(os.path.join(data_dir, outfile)):
        df = pd.read_csv(reviews_output, sep="\t", names=["userID", "itemID", "time"])
        df = filter_k_core(df, 10)  # filter for users & items with less than 10 interactions
        
        user_set, item_set = set(df['userID'].unique()), set(df['itemID'].unique())
        user_map = dict()
        item_map = dict()
        for u, user in enumerate(user_set):
            user_map[user] = u+1
        for i, item in enumerate(item_set):
            item_map[item] = i+1
        
        df["userID"] = df["userID"].apply(lambda x: user_map[x])
        df["itemID"] = df["itemID"].apply(lambda x: item_map[x])
        df = df.sort_values(by=["userID", "time"])
        df.drop(columns=["time"], inplace=True)
        df.to_csv(os.path.join(data_dir, outfile), sep="\t", header=False, index=False)

    inp_file = os.path.join(data_dir, dataset + ".txt")
    print('Using data from \n', inp_file)

    # initiate a dataset class 
    data = SASRecDataSet(filename=inp_file, col_sep="\t")
    # create train, validation and test splits
    data.split()

    # some statistics
    num_steps = int(len(data.user_train) / batch_size)
    cc = 0.0
    for u in data.user_train:
        cc += len(data.user_train[u])
    print('%g Users and %g items' % (data.usernum, data.itemnum))
    print('average sequence length: %.2f' % (cc / len(data.user_train)))

    if model_name == 'sasrec':
        model = SASREC(item_num=data.itemnum,
                    seq_max_len=maxlen,
                    num_blocks=num_blocks,
                    embedding_dim=hidden_units,
                    attention_dim=hidden_units,
                    attention_num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    conv_dims = [100, 100],
                    l2_reg=l2_emb,
                    num_neg_test=num_neg_test
        )
    elif model_name == "ssept":
        model = SSEPT(item_num=data.itemnum,
                    user_num=data.usernum,
                    seq_max_len=maxlen,
                    num_blocks=num_blocks,
                    # embedding_dim=hidden_units,  # optional
                    user_embedding_dim=10,
                    item_embedding_dim=hidden_units,
                    attention_dim=hidden_units,
                    attention_num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    conv_dims = [110, 110],
                    l2_reg=l2_emb,
                    num_neg_test=num_neg_test
        )
    else:
        print(f"Model-{model_name} not found")

    sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=3)

    with Timer() as train_time:
        t_test = model.train(data, sampler, num_epochs=num_epochs, batch_size=batch_size, lr=lr, val_epoch=6)
        res_syn = {"ndcg@10": t_test[0], "Hit@10": t_test[1]}
        print(res_syn)

    print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))