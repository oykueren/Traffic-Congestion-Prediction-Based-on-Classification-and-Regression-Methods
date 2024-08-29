import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.utils import check_array
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import warnings


class ST_DBSCAN():
    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 min_samples=5,
                 metric='euclidean',
                 n_jobs=-1):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        n, m = X.shape

        if len(X) < 20000:

            time_dist = pdist(X[:, 0].reshape(n, 1), metric=self.metric)
            euc_dist = pdist(X[:, 1:], metric=self.metric)

            # filter the euc_dist matrix using the time_dist
            dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)

            db = DBSCAN(eps=self.eps1,
                        min_samples=self.min_samples,
                        metric='precomputed')
            db.fit(squareform(dist))

            self.labels = db.labels_

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                nn_spatial = NearestNeighbors(metric=self.metric,
                                              radius=self.eps1)
                nn_spatial.fit(X[:, 1:])
                euc_sp = nn_spatial.radius_neighbors_graph(X[:, 1:],
                                                           mode='distance')

                nn_time = NearestNeighbors(metric=self.metric,
                                           radius=self.eps2)
                nn_time.fit(X[:, 0].reshape(n, 1))
                time_sp = nn_time.radius_neighbors_graph(X[:, 0].reshape(n, 1),
                                                         mode='distance')

                row = time_sp.nonzero()[0]
                column = time_sp.nonzero()[1]
                v = np.array(euc_sp[row, column])[0]

                dist_sp = coo_matrix((v, (row, column)), shape=(n, n))
                dist_sp = dist_sp.tocsc()
                dist_sp.eliminate_zeros()

                db = DBSCAN(eps=self.eps1,
                            min_samples=self.min_samples,
                            metric='precomputed')
                db.fit(dist_sp)
                self.labels = db.labels_
        return self

    def fit_frame_split(self, X, frame_size, frame_overlap=None):
        # check if input is correct
        X = check_array(X)

        # default values for overlap
        if frame_overlap == None:
            frame_overlap = self.eps2

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        if not frame_size > 0.0 or not frame_overlap > 0.0 or frame_size < frame_overlap:
            raise ValueError(
                'frame_size, frame_overlap not correctly configured.')

        # unique time points
        time = np.unique(X[:, 0])

        labels = None
        right_overlap = 0
        max_label = 0

        for i in range(0, len(time), (frame_size - frame_overlap + 1)):
            for period in [time[i:i + frame_size]]:
                frame = X[np.isin(X[:, 0], period)]

                self.fit(frame)

                # match the labels in the overlaped zone
                # objects in the second frame are relabeled
                # to match the cluster id from the first frame
                if not type(labels) is np.ndarray:
                    labels = self.labels
                else:
                    frame_one_overlap_labels = labels[len(labels) -
                                                      right_overlap:]
                    frame_two_overlap_labels = self.labels[0:right_overlap]

                    mapper = {}
                    for i in list(
                            zip(frame_one_overlap_labels,
                                frame_two_overlap_labels)):
                        mapper[i[1]] = i[0]
                    mapper[
                        -1] = -1  # avoiding outliers being mapped to cluster

                    # clusters without overlapping points are given new cluster
                    ignore_clusters = set(
                        self.labels) - set(frame_two_overlap_labels)
                    # recode them to new cluster value
                    if -1 in labels:
                        labels_counter = len(set(labels)) - 1
                    else:
                        labels_counter = len(set(labels))
                    for j in ignore_clusters:
                        mapper[j] = labels_counter
                        labels_counter += 1

                    new_labels = np.array([mapper[j] for j in self.labels])

                    # delete the right overlap
                    labels = labels[0:len(labels) - right_overlap]
                    # change the labels of the new clustering and concat
                    labels = np.concatenate((labels, new_labels))

                right_overlap = len(X[np.isin(X[:, 0],
                                              period[-frame_overlap + 1:])])
        self.labels = labels
        return self