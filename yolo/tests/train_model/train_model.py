from tests.train_model.settings import *
import matplotlib.pyplot as plt

yoloV3model = Model(
    yoloV3, loss_fn, optimizer, t_dataset, v_dataset,
    batch_size, device, scales, anchors, notify_after
)
yoloV3model.device

yoloV3model.fit(1)

for k, v in yoloV3model.history.items():
    print(k)
    print(v)


#--------------------------------------------------
# debug loss
#--------------------------------------------------
from model import YoloV3Loss
from utils import scale_anchors
from torch.nn import MSELoss

scaled_anchors = scale_anchors(anchors, 1, 640, 512)

im, tar = yoloV3model.t_dataset.__getitem__(0)
im = im.unsqueeze(0)
tar = tuple([t.unsqueeze(0) for t in tar])

pred = yoloV3model.model(im)

sloss = {}
for s, (ps, ts) in enumerate(zip(pred, tar)):
    sloss[s] = YoloV3Loss(device)(ps, ts, scaled_anchors[s: s+3])[1]

for i in [0, 1, 2]:
    sloss[i]


obj = tar[0][..., 4] == 1
no_obj = tar[0][..., 4] == 0

torch.zeros(pred[0][..., 0: 4][obj].shape).shape 

pred[0][..., 0: 4][no_obj].shape

tar[0][..., 0: 4][obj]

