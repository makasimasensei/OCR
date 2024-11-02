import numpy as np

from PostProcess.metric import metric
from Math.score import score


def score_fun(data, output_mynet):
    precision, recall, f1_score = [], [], []
    for i in range(len(output_mynet)):
        img_path = data['img_path'][i]
        box = data['box'][i]
        x = output_mynet[i]
        poly = metric(x, img_path, box)
        p, r, f = score(poly)
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1_score = np.mean(f1_score)
    return precision, recall, f1_score
