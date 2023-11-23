import numpy as np
import matplotlib.pyplot as plt
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
        # sorted_args = np.argsort(np.linalg.norm(cluster_centers, axis=1))[::-1]
        sorted_args = np.argsort(
            cluster_centers[:,0] * cluster_centers[:, 1]
        )[::-1]
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
