from tests.train_model.settings import *
from utils.scale_anchors import scale_anchors

yoloV3model = Model(
    yoloV3, loss_fn, optimizer, t_dataset, v_dataset,
    batch_size, device, scales, anchors, notify_after
)

anns = {}
for img_id in [0, 1, 2, 3, 4]:
    anns[img_id] = [x for x in annotations['annotations'] if x['image_id'] == img_id]

batch = []
for _batch in yoloV3model.t_dataloader:
    batch = _batch
    break
imgs, targets = batch[0], batch[1]

locs = {}
for img_id in range(5):
    locs[img_id] = []
    target = (targets[0][img_id], targets[1][img_id], targets[2][img_id])
    for scale in range(3):
        for anc_id, row, col in zip(*torch.where(target[scale][..., 4] == 1)):
            locs[img_id].append([
                torch.tensor([scale, anc_id, row, col]),
                target[scale][anc_id][row][col],
                anchors[anc_id]
            ])

locs[0][0]

for k, v in locs.items():
    print(k, len(v))
for k, v in anns.items():
    print(k, len(v))

for i in range(5):
    bboxs = [ann['bbox'] for ann in anns[i]]
    print(len(bboxs))
    bad_recon_bboxs = []
    for loc in locs[i]:
        (s, a, r, c), (x, y, w, h) = loc[0], loc[1][:4]
        recon_x  = (c + x) * scales[s]
        recon_y  = (r + y) * scales[s]
        recon_w = w * scales[s] 
        recon_h = h * scales[s]
        recon_bbox = torch.tensor([recon_x, recon_y, recon_w, recon_h])
        if list(recon_bbox) in bboxs:
            bboxs.remove(list(recon_bbox))
        else:
            bad_recon_bboxs.append(recon_bbox)
    print(len(bboxs), len(bad_recon_bboxs))
