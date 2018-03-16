#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:17:13 2018
DECISION TREE on lenses dataset
@author: shubham
"""
import os
import numpy as np
from decision_tree import DecisionTree
import data_manip as dm
CWD = os.getcwd() # Current Working Directory
lenses_path = CWD+os.sep+"lenses.txt"

def get_lenses_data (file_loc):
    with open(file_loc,'r') as f:
        dataset = []
        for lines in f.readlines():
            s = lines.strip("\n")
            s = s.split("  ")
            sample = []
            for i in range(1,len(s)):
                sample.append(int(s[i]))
            dataset.append(sample)
    dataset = np.array(dataset,dtype=np.float32)
    return dataset
#%%MAPS
NA_map = {0:"Age",1:"spectacle prescription",2:"Astigmatic",3:"Tear_production"}
AN_map = {"Age":0,"spectacle prescription":1,"Astigmatic":2,"Tear_production":3}
NV_map = {0:[1,2,3],#possible values of every attribute
          1:[1,2],
          2:[1,2],
          3:[1,2]}
NV_name_map ={0:{1:"young",2:"pre-presbyopic",3:"presbyopic"},
              1:{1:"myope",2:"hypermetrope"},
              2:{1:"No",2:"Yes"},
              3:{1:"Reduced",2:"Normal"}}


data = get_lenses_data(lenses_path)
print("Dataset Shape:",data.shape)
trees = []
correct = 0
N = 0
for fold in range(1,6):
    X = data[:,0:4]
    Y = data[:,4]
    P = dm.get_partitioned_dataset(X,Y,fold)

    X_train = P["train"]["X"]
    Y_train = P["train"]["Y"]
    X_val = P["val"]["X"]
    Y_val = P["val"]["Y"]

    print("=="*5,"Fold:",fold,"=="*5)
    print("Training Started.")
    tree = DecisionTree()
    tree.NV_name_map = NV_name_map
    tree.NA_map = NA_map
    tree.create_tree(X_train,Y_train,NV_map)
    print("Training Over.")
    print("Validating..")
    Y_pred = tree.predict(X_val)
    print("Actual Y:",Y_val)
    print("Predicted Y:",Y_pred)
    acc, crct, n = dm.get_accuracy(Y_val,Y_pred)
    print("Accuracy:",acc,"%")
    correct+=crct
    N+=n
    trees.append(tree)

print("Overall accuracy =",(correct*100)/N,"%")
print("Press 1 and Enter to print last tree.")
x = input()
if x == "1":
    tree.print_tree()
