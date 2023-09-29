import numpy as np
import matplotlib.pyplot as plt
import os
import tree_utils as utils
from sklearn.datasets import make_classification, make_blobs


clear = lambda : os.system('clear')


class Node():
    def __init__(
        self, 
        feature_index=None, 
        threshold=None, 
        left=None, 
        right=None, 
        info_gain=None, 
        value=None
    ):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value



class ClassificationTree:

    def __init__(self, min_samples, max_depth):
        self.min_samples = min_samples
        self.max_depth = max_depth 
        self.root = None


    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples >= self.min_samples and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = np.unique(Y)[
            np.array([*np.unique(Y, return_counts=True)])[1].argmax()
        ]
        # return leaf node
        return Node(value=leaf_value)


    def get_best_split(self, dataset, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.sort(np.unique(feature_values)).astype(float)
            ep = np.diff(possible_thresholds).min() / 3
            possible_thresholds += ep
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = utils.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = utils.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split


    def fit(self, X, y):
        dataset = np.hstack((X, y))
        self.root = self.build_tree(dataset)
    

    def predict(self, X):
        """
        Predict a sample set of features
        """

        if self.root is None:
            raise Exception("run fit before predict")

        preds = np.array([self.predit_single(x, self.root) for x in X])

        return preds


    def predit_single(self, x, tree):
        """
        Predict a single data point
        """
        if tree.value is not None:
            return tree.value
        else:
            if x[tree.feature_index] < tree.threshold:
                return self.predit_single(x, tree.left)
            else:
                return self.predit_single(x, tree.right)


    def print_tree(self, tree=None, indent=" "):
        """
        Print out the tree
        """

        if tree is None:
            tree = self.root
            
        if tree.value is not None:
            print (tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + indent)
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + indent)
