import cv2
import numpy as np
import pyclipper
from matplotlib import pyplot as plt

from shapely.geometry import Polygon

from PostProcess.findContours import find_contours


def metric(x, img_path, box):
    contours = find_contours(x, img_path)

    polygon_info = []
    # 在图像上绘制多边形
    for i in range(len(box)):
        temp = []
        if not contours:
            break
        for j, contour in enumerate(contours):
            # 将轮廓坐标转换为 (x, y) 格式
            contour_points = contour.squeeze()

            # 寻找包围多边形的最小矩形
            try:
                rect = cv2.minAreaRect(contour_points)
            except Exception:
                continue

            # 将最小矩形的信息转换为矩形的四个顶点坐标
            rect_box = cv2.boxPoints(rect)
            rect_box = np.intp(rect_box)

            poly = Polygon(rect_box)
            distance = poly.area * 1 / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(rect_box, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
            expanded = np.array(offset.Execute(distance))
            try:
                boxes = expanded.squeeze(axis=0)
            except:
                continue

            true_polygon = Polygon(box[i])
            predict_polygon = Polygon(boxes)

            try:
                intersection = true_polygon.intersection(predict_polygon)
                intersection_area = intersection.area
            except:
                continue

            if not temp:
                temp = [true_polygon, predict_polygon, intersection]
            else:
                if temp[2].is_empty:
                    temp_area = 0
                    if intersection_area > temp_area:
                        temp = [true_polygon, predict_polygon, intersection]
                else:
                    if intersection_area > temp[2].area:
                        temp = [true_polygon, predict_polygon, intersection]
        if temp:
            polygon_info.append(temp)
    return polygon_info
