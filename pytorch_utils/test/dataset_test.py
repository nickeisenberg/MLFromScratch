from pyaws.pytorch_utils import TreeImageFolder
import os

HOME = os.environ['HOME']

# quick test
root = f"{HOME}/GitRepos/pyaws_project/pyaws/_test/pytorch_utils/data"

dataset = TreeImageFolder(root=root, depth_limit=1)

dataset[0]
