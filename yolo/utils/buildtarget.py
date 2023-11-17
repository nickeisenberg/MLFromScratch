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
        sorted_args = np.argsort(np.linalg.norm(cluster_centers, axis=1))
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
    def __init__(self, anchors, scales, annotes):
        self.annotes = annotes
        self.anchors = anchors
        self.scales = scales
        self.anchor_assignment = {}
        self.ignore_keys = []
        self.target = (
            torch.zeros((1, 3, 16, 20, 6)),
            torch.zeros((1, 3, 32, 40, 6)),
            torch.zeros((1, 3, 64, 80, 6))
        )

    def best_anchor_for_annote(self, annote, ignore_keys=[]):
        """
        Given a flir annotation dictionary, this pair the dictionary with the
        anchor based on highest IOU score. When looping through all of the 
        annotations for a particular image, this function will replace an
        anchor assignment for a particular annotation if the new annotation has
        a higher IOU score, it will then reassign the previously assigned 
        annotation to a different anchor through a recursive processes.

        Parameters
        ----------
        annote: dict
            a flir annotation dictionary
        ignore_keys: list
            Not to be set by the user.
        """

        bbox = annote['bbox']
        best_iou = -1
        best_key = "0_0_0_0"
        for i, anchor in enumerate(self.anchors):
            anchor_scale = i // 3
            anchor_id = i % 3
            which_cell_row = (bbox[1] + (bbox[3] // 2)) // self.scales[anchor_scale]
            which_cell_col = (bbox[0] + (bbox[2] // 2)) // self.scales[anchor_scale]
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
                self.best_anchor_for_annote(replaced_annote, self.ignore_keys)
            else:
                self.ignore_keys.append(best_key)
                self.best_anchor_for_annote(annote, self.ignore_keys)

    def build_targets(self):
        for annote in self.annotes:
            self.best_anchor_for_annote(annote)
            self.ignore_keys = []
        for key, (annote, _) in self.anchor_assignment.items():
            sc, anchor, row, col = [int(x) for x in key.split("_")]
            self.target[sc][0, anchor, row, col] = torch.hstack([
                torch.Tensor(annote['bbox']), 
                torch.Tensor([1]), 
                torch.Tensor([annote['category_id']])
            ])


