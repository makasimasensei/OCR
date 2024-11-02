import numpy as np


def score(poly):
    precision, recall, f1_score = [], [], []
    for i in range(len(poly)):
        true_polygon_area = poly[i][0].area
        predict_polygon_area = poly[i][1].area
        intersection_area = poly[i][2].area

        # 真正类 TP
        tp = intersection_area

        # 假正类 FP
        fp = predict_polygon_area - tp

        # 假负类 FN
        fn = true_polygon_area - tp

        # Precision
        if tp + fp == 0:
            p = 0
        else:
            p = tp / (tp + fp)

        # Recall
        if tp + fn == 0:
            r = 0
        else:
            r = tp / (tp + fn)

        # F1 Score
        if p + r == 0:
            f = 0
        else:
            f = (2 * p * r) / (p + r)

        if p != 0 and r != 0 and f != 0:
            precision.append(p)
            recall.append(r)
            f1_score.append(f)

    if precision and recall and f1_score:
        return np.mean(precision), np.mean(recall), np.mean(f1_score)
    else:
        return 1, 1, 1
