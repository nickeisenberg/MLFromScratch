from sshtools.transfer import scp

src_root = "/home/ubuntu/GitRepos/ml_arcs/yolo/train_model"
dst_root = "/home/nicholas/GitRepos/ml_arcs/yolo/train_model"

files_to_move = [
    "/state_dicts/yolo_train1.pth",
    "/state_dicts/yolo_val1.pth",
    "/lossdfs/train1.csv",
    "/lossdfs/val1.csv",
]

user = "nicholas"
ip = "174.72.155.21"
port = "2201"

for file in files_to_move:
    scp(
