#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 00:11:55 2018
DECISION TREE
@author: shubham
"""
import math
import numpy as np
from copy import deepcopy

def lg(x):
    """log to the base 2"""
    return math.log(x,2)

class DecisionTree:
    def __init__(self):
        self.NA_map = None#Number->attribute name mapping
        self.AN_map = None#Attribute->Number mapping
        self.NV_map = None#Number-> list of possible values mapping
        self.NV_name_map = None#{attribute_number:{attribute_value:attribute_value_name}}
        self.X = None#input vectors
        self.Y = None#Output vectors
        self.root = None
        self.num_non_leaf_nodes = 0
        self.num_leaf_nodes = 0
        self.num_nodes = 0

    def create_tree(self, X=None, Y=None, NV_map = None):
        """A: Set of attributes"""
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if NV_map is None:
            NV_map = self.NV_map
        A = set()
        for key in NV_map.keys():
            A.add(key)
        N = X.shape[0]
        indices_list = list(range(N))
        self.root = Node()
        self._ID3(self.root,A,indices_list,X,Y,NV_map,1,1)

    def _ID3(self,node, A, indices_list, X, Y, NV_map,level,node_number):
        """A: Set of attributes to be tested.(Set of numbers)"""

        num_class, majority = self.find_majority(indices_list,Y)
        node.majority = majority
        c_ent = self.entropy_s([indices_list],Y)
        node.entropy = c_ent
        node.idx_list = indices_list
        node.node_id = "{"+str(level)+"}_{"+str(node_number)+"}"

        if num_class == 1:
            return
        if len(A) == 0:
            return

        a = self.select_best_attribute(A,indices_list,X,Y,NV_map,c_ent)
        partition = self.get_partition(a,NV_map,indices_list,X)
        node.best_attr = a

        node_attr_name=""
        if self.NA_map is not None:
            node_attr_name = self.NA_map[node.majority]
        node.node_id = node.node_id+node_attr_name

        A_new = deepcopy(A)
        A_new.discard(a)  # A_new <- A-{a}

        node_number = 1
        level = level + 1
        for a_val in partition.keys():
            ind_list = partition[a_val]
            if len(ind_list)==0:
                node.branches[a_val] = Node()
                node.branches[a_val].parent_id = node.node_id
                node.branches[a_val].majority = node.majority
                node.entropy = 0
                node.branches[a_val].node_id = "{"+str(level)+"}_{"+str(node_number)+"}"
            else:
                A_n = deepcopy(A_new)
                node.branches[a_val] = Node()
                node.branches[a_val].parent_id = node.node_id
                self._ID3(node.branches[a_val],A_n,ind_list,X,Y,NV_map,level,node_number)
            node_number+=1

    def find_majority (self,indices_list, Y):
        """Returns number of different classes and the label of the majority class"""
        hash_ = {}
        for i in indices_list:
            if Y[i] in hash_.keys():
                hash_[Y[i]] += 1
            else:
                hash_[Y[i]] = 1
        majority = None
        max_count = None

        for k in hash_.keys():
            if majority is None:
                majority = k
                max_count = hash_[k]
            else:
                if hash_[k]>max_count:
                    majority = k
                    max_count = hash_[k]
        return (len(hash_), majority)

    def get_partition (self,a,NV_map,indices_list,X):
        """Returns a dictionary of list. L= {a1:L1,a2:L2,..]
        Li is the list of indices where atrribute_value of a in X is NV_map[a][i].
        """
        L = {}
        for a_val in NV_map[a]:
            L[a_val] = []
        for i in indices_list:
            assert(X[i,a] in L.keys())
            L[X[i,a]].append(i)
        return L

    def select_best_attribute (self,A,indices_list,X,Y,NV_map,c_ent=None):
        """c_ent: current entropy"""
        i = 0
        if c_ent is None:
            c_ent = self.entropy_s([indices_list],Y)#current entropy
        for a in A:
            if i == 0:
                gain = c_ent-self.entropy_a(a,indices_list,NV_map,X,Y)
                max_gain = gain
                max_a = a
            else:
                gain = c_ent-self.entropy_a(a,indices_list,NV_map,X,Y)
                if gain>max_gain:
                    max_gain = gain
                    max_a = a

            i+=1
        return max_a

    def entropy_a (self,a,indices,NV_map=None,X=None,Y=None):
        if NV_map is None:
            assert(X is None and Y is None)
            NV_map = self.NV_map
            X = self.X
            Y = self.Y
        else:
            assert(not (X is None or Y is None))
        N = 0
        entropy = 0
        for a_value in NV_map[a]:
            n, e = self._entropy_a(a,a_value,indices,X,Y)
            N+=n
            entropy+= n*e
        return entropy/N

    def _entropy_a (self,a,a_val,indices,X,Y):
        n = 0
        count_map = {}
        for i in indices:
            if X[i,a] == a_val:
                n+=1
                if Y[i] in count_map.keys():
                    count_map[Y[i]] += 1
                else:
                    count_map[Y[i]] = 1
        entropy = 0
        for k in count_map.keys():
            p = (count_map[k])/n
            entropy-=(p*lg(p))
        return (n,entropy)

    def predict(self,x):
        if len(x.shape)==1:
            return self._predict(x)
        else:
            N = x.shape[0]
            Y = np.zeros((N,),dtype=x.dtype)
            for i in range(N):
                Y[i] = self._predict(x[i])
            return Y

    def _predict(self,x):
        result = [None]
        self._compute_output(self.root,x,result)
        return result[0]

    def _compute_output(self,node,x,result):
        """node: an instance of Node.
        x: input vector, result: a list having only one element(for passing by ref).
        """
        if node.is_leaf():
            result[0] = node.majority
            return
        else:
            next_node_val = x[node.best_attr]#value of the best_attr(in x) corresponfing
            # to the cuurent node.
            next_node = node.branches[next_node_val]
            self._compute_output(next_node,x,result)

    def entropy_s(self,indices_list,Y=None):
        """indices_list: list of list of indices"""
        if Y is None:
            Y = self.Y
        N = 0
        entropy = 0
        for indices in indices_list:
            n = len(indices)
            N = N + n
            entropy+=(n*self._entropy(indices,Y))
        return entropy/N

    def _entropy(self,indices,Y):
        """indices : list of indices"""
        N = len(indices)
        count_map = {}
        for i in indices:
            y = Y[i]
            if y in count_map.keys():
                count_map[y]+=1
            else:
                count_map[y]=1
        entropy = 0
        for label in count_map.keys():
            p = count_map[label]/N
            entropy = entropy - p*lg(p)
        return entropy

    def _print_tree(self,node):
        node.print_details()
        self.num_nodes+=1
        if node.is_leaf():
            self.num_leaf_nodes+=1

        if len(node.branches) == 0:
            return
        else:
            for val in node.branches.keys():
                if self.NV_name_map is not None:
                    name_value = self.NV_name_map[node.best_attr][val]
                    print("\n\nValue:({}) {}".format(val,name_value))
                else:
                    print("\n\nValue:",val)
                self._print_tree(node.branches[val])

    def print_tree(self):
        self.num_nodes = 0
        self.num_leaf_nodes = 0

        self._print_tree(self.root)
        print("Num of Total Nodes:",self.num_nodes)
        print("Num of Leaf Nodes:",self.num_leaf_nodes)

    def _count_nodes(self,node):
        self.num_nodes+=1
        if node.is_leaf():
            self.num_leaf_nodes+=1
            return
        else:
            for val in node.branches.keys():
                self._count_nodes(node.branches[val])

    def count_nodes(self,print_ = True):
        self.num_nodes = 0
        self.num_leaf_nodes = 0
        self._count_nodes(self.root)
        self.num_non_leaf_nodes = self.num_nodes-self.num_leaf_nodes
        if print_:
            print("Num of Total Nodes:",self.num_nodes)
            print("Num of Leaf Nodes:",self.num_leaf_nodes)
            print("Num of Non-Leaf Nodes:",self.num_non_leaf_nodes)

    def leafify(self,node_number,print_=True):
        """Makes node at node_number in Depth First Traversal a leaf node."""
        if print_:
            print("="*20)
            print("\nBefore Leafifying:")
        self.count_nodes(print_)

        if self.root.is_leaf():
            print("No Non-Leaf nodes to leafify")
            return
        nlf = self.num_non_leaf_nodes
        if node_number>(nlf):
            raise AssertionError("Node number greater than the num of non leaf nodes.")
        nn = [node_number]
        self._leafify(self.root,nn)
        if print_:
            print("="*20,"\nAfter Leafifying.\n")
        self.count_nodes(print_)

    def _leafify (self,root,node_number):
        if node_number[0] == 1:
            root.best_attr = None
            root.branches = []# may recursively delete to save memory
        else:
            for val in root.branches.keys():
                node = root.branches[val]
                if not node.is_leaf():
                    node_number[0]-=1
                    self._leafify(node,node_number)

class Node:
    def __init__(self):
        self.node_id = None
        self.parent_id = None
        self.majority = None#ouput value
        self.best_attr = None#Attribute
        self.dec_fn = None#decision function
        self.entropy = None
        self.branches = {}#children nodes. If None -> leaf
        self.idx_list = []#indices of the part of the dataset that corresponds to this node

    def is_leaf (self):
        if self.best_attr is None:
            return True
        return False

    def print_details(self):
        self._print_details()

    def _print_details(self):
        print("-"*10)
        print("Parent_id:",self.parent_id)
        print("Node_id:",self.node_id)
        print("Majority Class:", self.majority)
        print("Entropy:", self.entropy)
        print("idx_list:",self.idx_list)

if __name__ == "__main__":
    dt = DecisionTree()
    X = np.array([[0,0,0],[0,0,1],[0,1,0],
                  [0,1,1],[1,0,0],[1,0,1],
                  [1,1,0],[1,1,1]
                  ], dtype=np.float32)
    X2 = np.array([[0,0,0],[1,0,1],[0,1,0],
                  [1,1,1],[0,0,0],[1,0,1],
                  [1,1,0],[1,1,1]
                  ], dtype=np.float32)
    Y = np.array([0,1,0,1,0,1,0,1], dtype= np.float32)
#    Y = np.array([1,1,1,1,1,1,1,1], dtype= np.float32)
    print(dt.entropy_s([[0,2],[1,3]],[0,0,1,1]))
    idcs = [0,1,2,3,4,5,6,7]
    NV_map = {0:[0,1],1:[0,1],2:[0,1]}

    print(dt.entropy_a(0,idcs,NV_map,X,Y ))
    print(dt.select_best_attribute({0,1,2},idcs,X2,Y,NV_map))
    dt.create_tree(X2,Y,NV_map)
    dt.print_tree()