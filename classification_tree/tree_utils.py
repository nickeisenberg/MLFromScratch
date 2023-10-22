import numpy as np

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

def information_gain(
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
        gain = gini_index(parent)
        gain -= left_weight * gini_index(left)
        gain -= right_weight * gini_index(right) 
        return gain
    
    else:
        return -1 * np.inf




def split(dataset:np.ndarray, index:int, threshold:float) -> tuple:
    """
    Split a dataset into a left child and right child based on a feature 
    and threshold. feature < split is affirmative, feature >= split is the 
    alternative.
    """

    left_dataset = dataset[dataset[:, index] < threshold]
    right_dataset = dataset[dataset[:, index] >= threshold]
    return left_dataset, right_dataset
