import numpy as np
import os
clear = lambda : os.system('clear')

class ClassificationTree:

    def __init__(self, min_samples, max_depth):
        self.min_samples = min_samples
        self.max_depth = max_depth 


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
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split


    def information_gain(
        self, 
        parent:np.ndarray, 
        left:np.ndarray, 
        right:np.ndarray, 
        method='gini'
    ) -> float:

        """
        Find the information gain on a split from a parent group.
        """

        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        if method == 'gini':
            gain = self.gini_index(parent)
            gain -= left_weight * self.gini_index(left)
            gain -= right_weight * self.gini_index(right) 
            return gain
        
        else:
            return -1 * np.inf


    @staticmethod
    def gini_index(y):
    
        ''' 
        function to compute gini index
        '''
        
        class_labels = np.unique(y)
        gini = 0
        for label in class_labels:
            p_cls = len(y[y == label]) / len(y)
            gini += p_cls**2
    
        return 1 - gini


    @staticmethod
    def split(dataset:np.ndarray, index:int, threshold:float) -> tuple:
        """
        Split a dataset into a left child and right child based on a feature 
        and threshold. feature < split is affirmative, feature >= split is the 
        alternative.
        """

        left_dataset = dataset[dataset[:, index] < threshold]
        right_dataset = dataset[dataset[:, index] >= threshold]
        return left_dataset, right_dataset


dataset = np.vstack((np.arange(5), np.arange(5), np.array([1, 1, 2, 2, 2]))).T

dataset

dec = ClassificationTree(2, 2)

best = dec.get_best_split(dataset, 2)

best['dataset_left']
best['dataset_right']
