import torch
from torch.nn import Sigmoid
from .iou import iou
from .scale_anchors import scale_anchors

class BuildTarget:
    """
    A class to build the target for a single image.

    Parameters:
    ----------
    anchors: 1-D iterable
        A list of anchors in decreasing order based on area. As of now it is 
        assumed that there are 9 total anchors and 3 anchors per scale. The 
        anchors are to be scaled to full size.

    annotes: list
        A list of annotations for the image. Each annotation must be a 
        dictionary. There must must be a "bbox" key and a "category_id" id for
        the annotation. The bbox is of the form [x, y, w, h].
    """
    
    def __init__(self, category_mapper, anchors, scales, img_w, img_h):
        self.category_mapper = category_mapper
        self.category_mapper_inv = {v: k for k, v in category_mapper.items()}
        self.anchors = anchors
        self.full_scale_anchors = scale_anchors(anchors, 1, img_w, img_h)
        self.scales = scales 
        self.anchor_assignment = {}
        self.ignore_keys = []
        self.target = (
            torch.zeros((3, 16, 20, 6)),
            torch.zeros((3, 32, 40, 6)),
            torch.zeros((3, 64, 80, 6))
        )
        self.sigmoid = Sigmoid()

    def _best_anchor_for_annote(
            self, annote, ignore_keys=[], by_center=False
        ):
        """
        Given a flir annotation dictionary, this pair the dictionary with the
        anchor based on highest IOU score. When looping through all of the 
        annotations for a particular image, this function will replace an
        anchor assignment for a particular annotation if the new annotation has
        a higher IOU score, it will then reassign the previously assigned 
        annotation to a different anchor through a recursive processes.

        The function will store the information in self.anchor_assignment in
        the form self.anchor_assignment[(scale_id, anc_id, row, col)] = (annote, score)
        where row and col are the row and col number within the grid of scale_id.

        Parameters
        ----------
        annote: dict
            a flir annotation dictionary. It is assumed that the bbox is
            of the form (x, y, width, height).
        ignore_keys: list
            Not to be set by the user.
        by_center: bool default = False
            If true, then the bounding boxes are associated to a cell based off
            of the coordinates of their center. If false then it uses the
            coordinates of the upper left corner of the bounding box.
        """

        bbox = torch.tensor(annote['bbox'])
        best_iou = -1
        best_key = "0_0_0_0"
        for i, anchor in enumerate(self.full_scale_anchors):
            anchor_scale = i // 3
            anchor_id = i % 3

            if by_center:
                which_cell_row = (bbox[1] + (bbox[3] // 2)) // self.scales[anchor_scale]
                which_cell_col = (bbox[0] + (bbox[2] // 2)) // self.scales[anchor_scale]
            else:
                which_cell_row = bbox[1] // self.scales[anchor_scale]
                which_cell_col = bbox[0] // self.scales[anchor_scale]

            key = f"{anchor_scale}_{anchor_id}_{which_cell_row}_{which_cell_col}"

            if key in ignore_keys:
                continue

            _iou = iou(anchor, bbox, share_center=True)
            if _iou > best_iou:
                best_iou = _iou
                best_key = key

        if best_key not in self.anchor_assignment.keys():
            self.anchor_assignment[best_key] = (annote, best_iou)
            return None

        else:
            if best_iou > self.anchor_assignment[best_key][1]:
                replaced_annote = self.anchor_assignment[best_key][0]
                self.anchor_assignment[best_key] = (annote, best_iou) 
                self.ignore_keys.append(best_key)
                self._best_anchor_for_annote(replaced_annote, self.ignore_keys)

            else:
                self.ignore_keys.append(best_key)
                self._best_anchor_for_annote(annote, self.ignore_keys)

    def build_target(self, annotes, return_target=False, is_model_pred=True):
        """
        Loops through all annotations for an image and builds the target.
        The result will be added to self.target.

        Parameters
        ----------
        return_target: boolean, default=False
            If True, then this method will return self.target.

        target: boolean, default=False
            There is costomization in the way the loss function is defined.
            If False, then this function will set the target to be
            target[sc][anchor, row, col] = [bbox, 1, category_id]. However, it
            is starard to transform the bbox entries to the following:
                x, y = bbox[0] / self.scales[sc], bbox[1] / self.scales[sc]
                x, y = x - int(x), y - int(y)
                w, h = bbox[2] / self.scales[sc], bbox[3] / self.scales[sc]
                bbox = [x, y, w, h]
            If True, the this funtion will apply the above scaling.

        """

        for annote in annotes:
            self._best_anchor_for_annote(annote)
            self.ignore_keys = []

        for key, (annote, _) in self.anchor_assignment.items():
            sc, anchor_id, row, col = [int(x) for x in key.split("_")]
            bbox = annote['bbox']
            id = self.category_mapper[annote['category_id']]

            if is_model_pred:
                x, y = bbox[0] / self.scales[sc], bbox[1] / self.scales[sc]
                x, y = x - int(x), y - int(y)
                w, h = bbox[2] / self.scales[sc], bbox[3] / self.scales[sc]
                bbox = [x, y, w, h]

            self.target[sc][anchor_id, row, col] = torch.hstack([
                torch.Tensor(bbox), 
                torch.Tensor([1]), 
                torch.Tensor([id])
            ])

        if return_target:
            return self.target

        else:
            return None

    def decode_tuple(self, tup, p_thresh, iou_thresh, is_pred):
        if not is_pred:
            iou_thresh = 1

        all = []
        for scale_id, t in enumerate(tup):
            _all = []
            scale = self.scales[scale_id]
            scaled_ancs = scale_anchors(
                self.anchors[3 * scale_id: 3 * (scale_id + 1)], scale, 640, 512
            )
            dims = list(zip(*torch.where(t[..., 4:5] > p_thresh)[:-1]))
            for dim in dims:
                if is_pred:
                    bbox_info = t[dim][: 5]
                    bbox_info[:2] = self.sigmoid(bbox_info[:2])
                    bbox_info[4] = self.sigmoid(bbox_info[4])
                    x, y, w, h, p = bbox_info
                    cat = torch.argmax(t[dim][5:])
                    x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
                    w = torch.exp(w) * scaled_ancs[dim[0]][0] * scale
                    h = torch.exp(h) * scaled_ancs[dim[0]][1] * scale
                    cat = self.category_mapper_inv[cat.item()]
                else:
                    x, y, w, h, p, cat = t[dim]
                    x, y = (x + dim[2].item()) * scale, (y + dim[1].item()) * scale
                    w = w * scale
                    h = h * scale
                    cat = self.category_mapper_inv[cat.item()]
                _all.append(
                    {
                        'bbox': [x.item(), y.item(), w.item(), h.item()], 
                        'category_id': cat,
                        'p_score': p.item(),
                        'index': dim
                    }
                )
            all += _all
        
        sorted_all = sorted(all, key=lambda x: x['p_score'], reverse=True)
    
        keep = []
        while sorted_all:
            keep.append(sorted_all[0])
            _ = sorted_all.pop(0)
            for i, info in enumerate(sorted_all):
                score = iou(torch.tensor(keep[-1]['bbox']), torch.tensor(info['bbox']))
                if score > iou_thresh:
                    sorted_all.pop(i)
    
        return keep, all
