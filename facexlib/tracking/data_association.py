"""
For each detected item, it computes the intersection over union (IOU) w.r.t.
each tracked object. (IOU matrix)
Then, it applies the Hungarian algorithm (via linear_assignment) to assign each
det. item to the best possible tracked item (i.e. to the one with max IOU)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from facexlib.utils.misc import box_iou

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """Assigns detections to tracked object (both represented as bounding boxes)

    Returns:
        3 lists of matches, unmatched_detections and unmatched_trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = box_iou(detections, trackers)

    # The linear assignment module tries to minimize the total assignment cost.
    # In our case we pass -iou_matrix as we want to maximise the total IOU
    # between track predictions and the frame detection.
    row_ind, col_ind = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in row_ind:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in col_ind:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for row, col in zip(row_ind, col_ind):
        if iou_matrix[row, col] < iou_threshold:
            unmatched_detections.append(row)
            unmatched_trackers.append(col)
        else:
            matches.append(np.array([[row, col]]))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
