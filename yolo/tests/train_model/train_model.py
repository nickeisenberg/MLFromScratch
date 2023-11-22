from tests.train_model.settings import *

yoloV3model = Model(
    yoloV3, loss_fn, optimizer, dataset, train_val_split, 
    4, device, scales, anchors, 2
)

yoloV3model.fit(1)

for k, v in yoloV3model.history.items():
    print(k)
    print(v)
