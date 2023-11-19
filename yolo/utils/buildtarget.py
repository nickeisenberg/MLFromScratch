import torch
import numpy as np
import matplotlib.pyplot as plt
from .iou import iou
from sklearn.cluster import KMeans

class ConstructAnchors:
    def __init__(self, annotations, img_width, img_height):
        self.annotations = annotations
        self.bboxes = np.array([
            [x['bbox'][2] / img_width, x['bbox'][3] / img_height]  
            for x in annotations
        ])
        self.k_means()

    def k_means(self, n_clusters=9):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.clusters = self.kmeans.fit_predict(self.bboxes)
        cluster_centers = self.kmeans.cluster_centers_
        sorted_args = np.argsort(np.linalg.norm(cluster_centers, axis=1))[::-1]
        self.cluster_centers = np.hstack(
            (sorted_args.reshape((-1, 1)), cluster_centers[sorted_args])
        )
        return None

    def view_clusters(self, show=True):
        fig = plt.figure()
        plt.scatter(self.bboxes[:, 0], self.bboxes[:, 1], c=self.clusters)
        if show:
            plt.show()
        else:
            return fig


class BuildTarget:
    def __init__(self, anchors, annotes, scales, img_size):
        self.annotes = annotes
        self.anchors = anchors
        self.scales = scales
        self.img_size = img_size
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

        bbox = annote['bbox']
        best_iou = -1
        best_key = "0_0_0_0"
        for i, anchor in enumerate(self.anchors):
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
            


        match_bbox_to_pred will transform the bbox to match the model output. I
        may just set this as a permenant default. For now it is an option set
        to True.

        """
        for annote in self.annotes:
            self._best_anchor_for_annote(annote)
            self.ignore_keys = []

        for key, (annote, _) in self.anchor_assignment.items():
            sc, anchor, row, col = [int(x) for x in key.split("_")]
            bbox = annote['bbox']

            if match_bbox_to_pred:
                x, y = bbox[0] / self.scales[sc], bbox[1] / self.scales[sc]
                x, y = x - int(x), y - int(y)
                w, h = bbox[2] / self.scales[sc], bbox[3] / self.scales[sc]
                bbox = [x, y, w, h]

            self.target[sc][anchor, row, col] = torch.hstack([
                torch.Tensor(bbox), 
                torch.Tensor([1]), 
                torch.Tensor([annote['category_id']])
            ])

        if return_target:
            return self.target

        else:
            return None



