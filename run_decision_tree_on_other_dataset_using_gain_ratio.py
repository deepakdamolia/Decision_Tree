#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:17:13 2018
DECISION TREE on other dataset
@author: shubham
"""
import os
import numpy as np
from copy import deepcopy
from decision_tree_gain_ratio import DecisionTree, Node
import data_manip as dm
import string
CWD = os.getcwd() # Current Working Directory
np.random.seed(0)

#%%DATA
train_path =  CWD + os.sep + "training_set.csv"
val_path =  CWD + os.sep + "validation_set.csv"
test_path =  CWD + os.sep + "test_set.csv"
d_train = np.genfromtxt(train_path,delimiter=',')
d_train = (d_train[1:]).astype(np.float32)
d_val = np.genfromtxt(val_path,delimiter=',')
d_val = d_val[1:].astype(np.float32)
d_test = np.genfromtxt(test_path,delimiter=',')
d_test = d_test[1:].astype(np.float32)
print("Data:")
print("Train:",d_train.shape)
print("Validation:",d_val.shape)
print("Test:",d_test.shape)

X_train = d_train[:, 0:20]  # first 20
Y_train = d_train[:, 20]  # last element
X_val = d_val[:, 0:20]  # first 20
Y_val = d_val[:, 20]  # last element
X_test = d_test[:, 0:20]  # first 20
Y_test = d_test[:, 20]  # last element

#%%MAPS
NV_map = {}
alphabets = list(string.ascii_uppercase)
NA_map = {}
for i in range(20):  # 0-19
    A = "X"+alphabets[i+1]
    NV_map[i] = [0,1]# because only 0 and 1 are possible values for every attaribute
    NA_map[i] = A

print(NV_map)
print(NA_map)



#%%TREE
tree = DecisionTree()
tree.NA_map = NA_map
print("Creating Tree....")
tree.create_tree(X_train,Y_train,NV_map)
print("Decision Tree created.")
flag = input("Press 1/0 and Enter to print/not-print the tree.")

if (flag != ""):
    if int(flag)==1:
        tree.print_tree()
        msg = "NODE_ID_FORMAT: {level}_{branch_number} Name(if available)"
        print("="*len(msg))
        print(msg)
        print("="*len(msg))
Y_p = tree.predict(X_train)
acc, _, _ = dm.get_accuracy(Y_train,Y_p)
print("Training accuracy:",acc,"%")

Y_p = tree.predict(X_val)
acc, _, _ = dm.get_accuracy(Y_val,Y_p)
print("Validation accuracy:",acc,"%")

Y_p = tree.predict(X_test)
acc, _, _ = dm.get_accuracy(Y_test,Y_p)
print("Test accuracy:",acc,"%")

#%%PRUNING
dummy = input("\n\nPress Enter for Pruning")

def get_best_pruned_tree (D,L,K,X_val,Y_val,verbose=True):
    """Return best pruned tree"""
    Dbest = D
    Ddash = None
    best_acc, _ , _ = dm.get_accuracy(D.predict(X_val),Y_val)
    logs = []
    for i in range(L):
        Ddash = deepcopy(D)
        M = np.random.randint(1,K+1)# rand num in [1,K]
        for m in range(M):
            Ddash.count_nodes(print_=False)
            N = Ddash.num_non_leaf_nodes
            if N == 0:
                break
            n = np.random.randint(1,N+1)
            Ddash.leafify(n,print_=False)  #make nth non-leaf node a leaf
        acc, _, _ = dm.get_accuracy(Ddash.predict(X_val),Y_val)
#        print("\nCurrent Tree Accuracy:",acc,"%")
        if acc>best_acc:
            best_acc = acc
            Dbest = Ddash
        else:
            del Ddash
#        print("Current Best Accuracy:",best_acc,"%")
        log = [i+1,M,acc,best_acc]
        logs.append(log)
    if verbose:
        for log in logs:
            msg = "sr.no.={} , M = {}, Accuracy = {}%, Best = {}%".format(log[0],
                          log[1],log[2],log[3])
            print("="*len(msg),"\n",msg,"\n")
        msg = "Final Best Tree Accuracy: "+str(best_acc)+" %"
        print("="*len(msg),"\n",msg,"\n","="*len(msg))
    return (Dbest, best_acc)


#set of Ls and Ks to be used. Total 10 combinations
Ls = [15,20]
Ks = [15,20,30,40,50]

val_acc,_,_ = dm.get_accuracy(Y_val,tree.predict(X_val))
test_acc,_,_ = dm.get_accuracy(Y_test,tree.predict(X_test))
print("Initial Accuracies:")
print("Validation Accuracy:",val_acc,"%")
print("Test Accuracy",test_acc,"%")

srno = 0
Dbest = tree
best_val_acc = val_acc
best_L = 0
best_K = 0
for L in Ls:
    for K in Ks:
        Dnew, new_val_acc = get_best_pruned_tree(tree,L,K,X_val,Y_val,False)

        if new_val_acc>best_val_acc:
            best_val_acc = new_val_acc
            Dbest = Dnew
            best_L = L
            best_K = K
        else:
            del Dnew
        srno+=1
        msg = "sr.no.={}, L={}, K={}, Validation \
Accuracy = {}%, Best = {}%".format(srno,L,K,new_val_acc,best_val_acc)
        print("="*len(msg),"\n",msg,"\n")

Y_pred = Dbest.predict(X_test)
test_accuracy = dm.get_accuracy(Y_test,Y_pred)

msg = "Test accuracy of the best tree(L={},K={})={}%".format(best_L,best_K,test_accuracy[0])
print("="*len(msg),"\n",msg,"\n")
print("{} out of {} predictions are correct".format(test_accuracy[1],test_accuracy[2]))