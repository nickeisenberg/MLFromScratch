import torch
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
    
    def __init__(self, anchors, annotes, scales, img_w, img_h):
        self.annotes = annotes
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

    def build_targets(self, return_target=False, match_bbox_to_pred=False):
        """
        Loops through all annotations for an image and builds the targets.
        The result will be added to self.target.

        Parameters
        ----------
        return_target: boolean, default=False
            If True, then this method will return self.target.

        match_bbox_to_pred: boolean, default=False
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

        for annote in self.annotes:
            self._best_anchor_for_annote(annote)
            self.ignore_keys = []

        for key, (annote, _) in self.anchor_assignment.items():
            sc, anchor_id, row, col = [int(x) for x in key.split("_")]
            bbox = annote['bbox']

            if match_bbox_to_pred:
                x, y = bbox[0] / self.scales[sc], bbox[1] / self.scales[sc]
                x, y = x - int(x), y - int(y)
                w, h = bbox[2] / self.scales[sc], bbox[3] / self.scales[sc]
                bbox = [x, y, w, h]

            self.target[sc][anchor_id, row, col] = torch.hstack([
                torch.Tensor(bbox), 
                torch.Tensor([1]), 
                torch.Tensor([annote['category_id']])
            ])

        if return_target:
            return self.target

        else:
            return None
