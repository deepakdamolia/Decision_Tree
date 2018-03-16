#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:19:20 2018
For Data manioulation/ordering etc.
@author: shubham
"""
import numpy as np
#For 5-fold cross validation
def get_chunk_ranges (range_ ,fold_number):
    a = range_[0]
    b = range_[1]
    sz = b - a + 1
    rem = sz%5
    cs = int (sz/5)
    train = []
    val = []
    j = a
    for i in range(1,6):
        csz = cs #chunk size
        if rem != 0:
            rem-=1
            csz+=1
        r= (j, j + csz -1)  # as r[1] is inclusive
        if i==fold_number:
            val.append(r)
        else:
            train.append(r)
        j = j + csz
    return {"train":train,"val":val}

def get_split_ranges (ranges,fold_number):
    train = []
    val = []
    for item in ranges:
        split = get_chunk_ranges(item,fold_number)
        for r in split["train"]:
            train.append(r)
        for r in split["val"]:
            val.append(r)
    return {"train":train,"val":val}

def fetch_concatenated_chunks (X,Y,ranges):
    Lx = []
    Ly = []
    for r in ranges:
        a = r[0]
        b = r[1]+1
        Lx.append(X[a:b])
        Ly.append(Y[a:b])
    Lx = np.concatenate(Lx,axis=0)
    Ly = np.concatenate(Ly,axis=0)
    return (Lx,Ly)

def get_partitioned_dataset (X,Y,fold_number):
    N = X.shape[0]
    parts = get_split_ranges([[0,N-1]],fold_number)
    partition = {}
    x_train , y_train = fetch_concatenated_chunks (X,Y,parts["train"])
    partition["train"] = {"X":x_train,"Y":y_train}
    x_val , y_val = fetch_concatenated_chunks (X,Y,parts["val"])
    partition["val"] = {"X":x_val,"Y":y_val}
    return partition

def get_accuracy (y_true, y_pred):
    N  = y_true.shape[0]
    assert(N==y_pred.shape[0] and (y_true.dtype==y_pred.dtype))
    mask = (y_true==y_pred)
    flags = np.zeros((N,))
    flags[mask==True] = 1.0
    correct = np.sum(flags)
    return ((correct/N)*100,correct,N)


if __name__ == "__main__":
    for i in range(5):
        print(i+1,get_split_ranges([[0,23]],i+1) )