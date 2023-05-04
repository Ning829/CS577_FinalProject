import pandas as pd
import numpy as np
import os
import io
import string
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import jieba    # for tokenization
import paddle

from gensim.models import KeyedVectors
from statistics import mean


def embedding(sentence, keyVector, max_len):
    vector_list = []
    dict = keyVector.key_to_index

    # add word vector
    for word in sentence:
        if word in dict:
            vector_list.append(keyVector.get_vector(word))
        else:
            vector_list.append(np.zeros(100))

    # change to tensor
    vector_list = np.array(vector_list)
    vector_list = torch.tensor(vector_list)

    # pad length
    vector_list = F.pad(vector_list, (0, 0, 0, max_len - len(vector_list)), "constant", 0)
    return vector_list


def one_hot(y, c):
    y_onehot = np.zeros((y.shape[0], c))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


class WeiboDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DAN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DAN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X1):
        X1 = torch.mean(X1, dim=1)

        X1 = self.fc1(X1)

        X1 = torch.relu(X1)

        X = self.fc2(X1)
        X = self.sigmoid(X)
        return X


if __name__ == '__main__':
    # read files
    df = pd.read_csv('data/train.csv')
    df = df[df['原始图片url'] == "无"]

    test_df = pd.read_csv('data/test.csv')
    test_df = test_df[test_df['原始图片url'] == "无"]

    # remove bad data
    discard = ["弱智吧每周最佳", "【", "青龙山皇家疗养院", "吴亦凡被刑拘"]
    df = df[~df.微博正文.str.contains('|'.join(discard))]
    df = df[df.微博正文.str.contains("——")]

    test_df = test_df[~test_df.微博正文.str.contains('|'.join(discard))]
    test_df = test_df[test_df.微博正文.str.contains("——")]

    # remove everything following "——" in each text
    for i in df.index:
        df.loc[i, '微博正文'] = df.loc[i, '微博正文'][:df.loc[i, '微博正文'].rindex("——")]

    for i in test_df.index:
        test_df.loc[i, '微博正文'] = test_df.loc[i, '微博正文'][:test_df.loc[i, '微博正文'].rindex("——")]

    # tokenization
    paddle.enable_static()
    jieba.enable_paddle()

    sentences = df.微博正文
    sentences_seg = []
    for sentence in sentences:
        seg_list = jieba.cut(sentence, use_paddle=True)     # use paddle
        word_list = list(seg_list)
        sentences_seg.append(word_list)

    test_sentences = test_df.微博正文
    test_sentences_seg = []
    for sentence in test_sentences:
        seg_list = jieba.cut(sentence, use_paddle=True)
        word_list = list(seg_list)
        test_sentences_seg.append(word_list)

    # embedding
    file_path = "tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt"
    tx_embedding = KeyedVectors.load_word2vec_format(file_path, binary=False, limit=50000)

    max_length = 0
    for sentence in sentences_seg:
        if len(sentence) > max_length:
            max_length = len(sentence)

    embedding_list = []
    for sentence in sentences_seg:
        embedding_list.append(embedding(sentence, tx_embedding, max_length))

    test_embedding_list = []
    for sentence in test_sentences_seg:
        test_embedding_list.append(embedding(sentence, tx_embedding, max_length))

    # process y
    likes_number = df.点赞数.astype(int)

    q1 = likes_number.quantile(q=0.25, interpolation="higher")
    q2 = likes_number.quantile(q=0.5, interpolation="higher")
    q3 = likes_number.quantile(q=0.75, interpolation="higher")

    for i in likes_number.index:
        if likes_number.loc[i] <= q1:
            likes_number.loc[i] = 0
        elif likes_number.loc[i] <= q2:
            likes_number.loc[i] = 1
        elif likes_number.loc[i] <= q3:
            likes_number.loc[i] = 2
        else:
            likes_number.loc[i] = 3

    onehot_likes = one_hot(likes_number, 4)

    test_likes_number = test_df.点赞数.astype(int)
    for i in test_likes_number.index:
        if test_likes_number.loc[i] <= q1:
            test_likes_number.loc[i] = 0
        elif test_likes_number.loc[i] <= q2:
            test_likes_number.loc[i] = 1
        elif test_likes_number.loc[i] <= q3:
            test_likes_number.loc[i] = 2
        else:
            test_likes_number.loc[i] = 3

    test_onehot_like = one_hot(test_likes_number, 4)

    # divide into 5 sets
    size = round(len(onehot_likes)/5)
    Xsubset = []
    ysubset = []
    for i in range(5):
        Xsubset.append(embedding_list[size*i:size * (i+1)])
        ysubset.append(onehot_likes[size*i:size * (i+1)])

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for X, y in zip(Xsubset, ysubset):
        X_train.append(X[:round(size * 0.8)])
        y_train.append(y[:round(size * 0.8)])
        X_val.append(X[round(size * 0.8):])
        y_val.append(y[round(size * 0.8):])

    a_train = WeiboDataset(X_train[0], y_train[0])
    a_train_loader = torch.utils.data.DataLoader(a_train, batch_size=32, shuffle=True)
    b_train = WeiboDataset(X_train[1], y_train[1])
    b_train_loader = torch.utils.data.DataLoader(b_train, batch_size=32, shuffle=True)
    c_train = WeiboDataset(X_train[2], y_train[2])
    c_train_loader = torch.utils.data.DataLoader(c_train, batch_size=32, shuffle=True)
    d_train = WeiboDataset(X_train[3], y_train[3])
    d_train_loader = torch.utils.data.DataLoader(d_train, batch_size=32, shuffle=True)
    e_train = WeiboDataset(X_train[4], y_train[4])
    e_train_loader = torch.utils.data.DataLoader(e_train, batch_size=32, shuffle=True)

    a_val = WeiboDataset(X_val[0], y_val[0])
    a_val_loader = torch.utils.data.DataLoader(a_val, batch_size=32, shuffle=True)
    b_val = WeiboDataset(X_val[1], y_val[1])
    b_val_loader = torch.utils.data.DataLoader(b_val, batch_size=32, shuffle=True)
    c_val = WeiboDataset(X_val[2], y_val[2])
    c_val_loader = torch.utils.data.DataLoader(c_val, batch_size=32, shuffle=True)
    d_val = WeiboDataset(X_val[3], y_val[3])
    d_val_loader = torch.utils.data.DataLoader(d_val, batch_size=32, shuffle=True)
    e_val = WeiboDataset(X_val[4], y_val[4])
    e_val_loader = torch.utils.data.DataLoader(e_val, batch_size=32, shuffle=True)

    # training parameters
    max_epoch = 500
    input_size = 100

    model_a = DAN(input_size, hidden_size=200, output_size=4)
    model_b = DAN(input_size, hidden_size=50, output_size=4)
    model_c = DAN(input_size, hidden_size=200, output_size=4)
    model_d = DAN(input_size, hidden_size=100, output_size=4)
    model_e = DAN(input_size, hidden_size=100, output_size=4)

    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.005)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.015)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=0.011)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.003)
    optimizer_e = torch.optim.Adam(model_e.parameters(), lr=0.013)

    criterion = torch.nn.CrossEntropyLoss()

    train_loss_a = []
    train_loss_b = []
    train_loss_c = []
    train_loss_d = []
    train_loss_e = []
    train_accuracy_a = []
    train_accuracy_b = []
    train_accuracy_c = []
    train_accuracy_d = []
    train_accuracy_e = []
    validation_accuracy_a = []
    validation_accuracy_b = []
    validation_accuracy_c = []
    validation_accuracy_d = []
    validation_accuracy_e = []

    # training loop
    for epoch in range(max_epoch):
        ################################  A  ###############################
        epoch_loss = 0
        num_correct_train = 0
        for X, y in a_train_loader:
            optimizer_a.zero_grad()
            X = X.to(torch.float32)
            outputs = model_a(X)

            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer_a.step()

            epoch_loss += loss.item()
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_train += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        # evaluation mode
        model_a.eval()

        num_correct_val = 0
        for X,y in a_val_loader:
            X = X.to(torch.float32)
            outputs = model_a(X)
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_val += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        train_loss_a.append(epoch_loss / len(a_train_loader))
        train_accuracy_a.append(num_correct_train / len(a_train_loader.dataset))
        validation_accuracy_a.append(num_correct_val / len(a_val_loader.dataset))

        print(f"Model A : Epoch {epoch + 1}: loss = {epoch_loss / len(a_train_loader)}, "
              f"train accu = {num_correct_train / len(a_train_loader.dataset)}, "
              f"val accu = {num_correct_val / len(a_val_loader.dataset)}, ")

        ################################  B  ###############################
        epoch_loss = 0
        num_correct_train = 0
        for X, y in b_train_loader:
            optimizer_b.zero_grad()
            X = X.to(torch.float32)
            outputs = model_b(X)

            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer_b.step()

            epoch_loss += loss.item()
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_train += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        # evaluation mode
        model_b.eval()

        num_correct_val = 0
        for X,y in b_val_loader:
            X = X.to(torch.float32)
            outputs = model_b(X)
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_val += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        train_loss_b.append(epoch_loss / len(b_train_loader))
        train_accuracy_b.append(num_correct_train / len(b_train_loader.dataset))
        validation_accuracy_b.append(num_correct_val / len(b_val_loader.dataset))

        print(f"Model B : Epoch {epoch + 1}: loss = {epoch_loss / len(b_train_loader)}, "
              f"train accu = {num_correct_train / len(b_train_loader.dataset)}, "
              f"val accu = {num_correct_val / len(b_val_loader.dataset)}, ")

        ################################  C  ###############################
        epoch_loss = 0
        num_correct_train = 0
        for X, y in c_train_loader:
            optimizer_c.zero_grad()
            X = X.to(torch.float32)
            outputs = model_c(X)

            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer_c.step()

            epoch_loss += loss.item()
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_train += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        # evaluation mode
        model_c.eval()

        num_correct_val = 0
        for X,y in c_val_loader:
            X = X.to(torch.float32)
            outputs = model_c(X)
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_val += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        train_loss_c.append(epoch_loss / len(c_train_loader))
        train_accuracy_c.append(num_correct_train / len(c_train_loader.dataset))
        validation_accuracy_c.append(num_correct_val / len(c_val_loader.dataset))

        print(f"Model C : Epoch {epoch + 1}: loss = {epoch_loss / len(c_train_loader)}, "
              f"train accu = {num_correct_train / len(c_train_loader.dataset)}, "
              f"val accu = {num_correct_val / len(c_val_loader.dataset)}, ")

        ################################  D  ###############################
        epoch_loss = 0
        num_correct_train = 0
        for X, y in d_train_loader:
            optimizer_d.zero_grad()
            X = X.to(torch.float32)
            outputs = model_d(X)

            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer_d.step()

            epoch_loss += loss.item()
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_train += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        # evaluation mode
        model_d.eval()

        num_correct_val = 0
        for X,y in d_val_loader:
            X = X.to(torch.float32)
            outputs = model_d(X)
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_val += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        train_loss_d.append(epoch_loss / len(d_train_loader))
        train_accuracy_d.append(num_correct_train / len(d_train_loader.dataset))
        validation_accuracy_d.append(num_correct_val / len(d_val_loader.dataset))

        print(f"Model D : Epoch {epoch + 1}: loss = {epoch_loss / len(d_train_loader)}, "
              f"train accu = {num_correct_train / len(d_train_loader.dataset)}, "
              f"val accu = {num_correct_val / len(d_val_loader.dataset)}, ")

        ################################  E  ###############################
        epoch_loss = 0
        num_correct_train = 0
        for X, y in e_train_loader:
            optimizer_e.zero_grad()
            X = X.to(torch.float32)
            outputs = model_e(X)

            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer_e.step()

            epoch_loss += loss.item()
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_train += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        # evaluation mode
        model_e.eval()

        num_correct_val = 0
        for X,y in e_val_loader:
            X = X.to(torch.float32)
            outputs = model_e(X)
            predict = outputs.detach()
            index = np.argmax(predict, axis=1)
            num_correct_val += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

        train_loss_e.append(epoch_loss / len(e_train_loader))
        train_accuracy_e.append(num_correct_train / len(e_train_loader.dataset))
        validation_accuracy_e.append(num_correct_val / len(e_val_loader.dataset))

        print(f"Model E : Epoch {epoch + 1}: loss = {epoch_loss / len(e_train_loader)}, "
              f"train accu = {num_correct_train / len(e_train_loader.dataset)}, "
              f"val accu = {num_correct_val / len(e_val_loader.dataset)}, ")

    # hyperparameter tuning
    """
    lr_accu = []
    for lr in np.arange(0.0005, 0.007, 0.001):
        print(lr)
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr)
        for epoch in range(max_epoch):
            ################################  A  ###############################
            epoch_loss = 0
            num_correct_train = 0
            for X, y in a_train_loader:
                optimizer_a.zero_grad()
                X = X.to(torch.float32)
                outputs = model_a(X)

                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer_a.step()

                epoch_loss += loss.item()
                predict = outputs.detach()
                index = np.argmax(predict, axis=1)
                num_correct_train += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

            # evaluation mode
            model_a.eval()

            num_correct_val = 0
            for X,y in a_val_loader:
                X = X.to(torch.float32)
                outputs = model_a(X)
                predict = outputs.detach()
                index = np.argmax(predict, axis=1)
                num_correct_val += np.sum(np.array(index) == np.array(np.argmax(y, axis=1)))

            train_loss_a.append(epoch_loss / len(a_train_loader))
            train_accuracy_a.append(num_correct_train / len(a_train_loader.dataset))
            validation_accuracy_a.append(num_correct_val / len(a_val_loader.dataset))

            print(f"Model A : Epoch {epoch + 1}: loss = {epoch_loss / len(a_train_loader)}, "
                  f"train accu = {num_correct_train / len(a_train_loader.dataset)}, "
                  f"val accu = {num_correct_val / len(a_val_loader.dataset)}, ")

        lr_accu.append(mean(validation_accuracy_a))

    # plot lr vs accuracy
    plt.clf()
    plt.plot(np.arange(0.0005, 0.007, 0.001), lr_accu)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Learning Rate vs Accuracy")
    plt.savefig("lr_vs_accu.png")
    """

    # plot learning curves
    """
    # A
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(train_loss_a, label="train loss", color='r')
    ax.legend(loc='upper left')

    ax2.plot(train_accuracy_a, label="train accuracy")
    ax2.plot(validation_accuracy_a, label="val accuracy")
    ax2.legend(loc='upper right')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='r')
    ax2.set_ylabel("Accuracy")
    plt.title("Learning Curves for Subset A")
    plt.tight_layout()
    plt.savefig("learning_curve_a.png")

    # B
    plt.clf()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(train_loss_b, label="train loss", color='r')
    ax.legend(loc='upper left')

    ax2.plot(train_accuracy_b, label="train accuracy")
    ax2.plot(validation_accuracy_b, label="val accuracy")
    ax2.legend(loc='upper right')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='r')
    ax2.set_ylabel("Accuracy")
    plt.title("Learning Curves for Subset B")
    plt.tight_layout()
    plt.savefig("learning_curve_b.png")

    # C
    plt.clf()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(train_loss_c, label="train loss", color='r')
    ax.legend(loc='upper left')

    ax2.plot(train_accuracy_c, label="train accuracy")
    ax2.plot(validation_accuracy_c, label="val accuracy")
    ax2.legend(loc='upper right')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='r')
    ax2.set_ylabel("Accuracy")
    plt.title("Learning Curves for Subset C")
    plt.tight_layout()
    plt.savefig("learning_curve_c.png")

    # D
    plt.clf()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(train_loss_d, label="train loss", color='r')
    ax.legend(loc='upper left')

    ax2.plot(train_accuracy_d, label="train accuracy")
    ax2.plot(validation_accuracy_d, label="val accuracy")
    ax2.legend(loc='upper right')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='r')
    ax2.set_ylabel("Accuracy")
    plt.title("Learning Curves for Subset D")
    plt.tight_layout()
    plt.savefig("learning_curve_d.png")

    # E
    plt.clf()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(train_loss_e, label="train loss", color='r')
    ax.legend(loc='upper left')

    ax2.plot(train_accuracy_e, label="train accuracy")
    ax2.plot(validation_accuracy_e, label="val accuracy")
    ax2.legend(loc='upper right')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color='r')
    ax2.set_ylabel("Accuracy")
    plt.title("Learning Curves for Subset E")
    plt.tight_layout()
    plt.savefig("learning_curve_e.png")
    """

    # predict
    test_embedding_list = torch.stack(test_embedding_list)
    test_embedding_list = test_embedding_list.to(torch.float32)

    a_output = model_a(test_embedding_list)
    b_output = model_b(test_embedding_list)
    c_output = model_c(test_embedding_list)
    d_output = model_d(test_embedding_list)
    e_output = model_e(test_embedding_list)

    final_output = 2 * a_output + 0.005 * b_output + 0.01 * c_output + 0.01 * d_output + 0.005 * e_output

    predict = final_output.detach()
    index = np.argmax(predict, axis=1)

    num_correct_test = 0
    num_correct_test += np.sum(np.array(index) == np.array(np.argmax(test_onehot_like, axis=1)))
    test_acc = num_correct_test / test_onehot_like.shape[0]

    print(test_acc)
