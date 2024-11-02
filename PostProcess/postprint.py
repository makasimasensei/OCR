import os

import cv2
import numpy as np
import pyclipper
import torch
from matplotlib import pyplot as plt
from shapely import Polygon
from PIL import Image, ImageDraw


def post_print(x, input_pic, t):
    img = cv2.imread(input_pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    original = torch.permute(x, (1, 2, 0))
    original = original.cpu().detach().numpy()
    original = cv2.resize(original, (w, h))

    image = x[0].cpu().detach().numpy()
    _, binary_image = cv2.threshold(image, 0.3, 1, cv2.THRESH_BINARY)
    binary_image = cv2.resize(binary_image, (w, h))
    binary_image = np.uint8(binary_image)

    # 使用cv2.connectedComponentsWithStats获取连通组件信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    # 输出每个连通组件的面积（注意：第一个标签是背景，所以从1开始）
    areas = stats[1:, cv2.CC_STAT_AREA]

    # 找到面积为150的组件的标签
    small_area_labels = np.where(areas <= 50)[0] + 1  # 加1是因为标签从1开始

    # 将面积为150的像素置零
    for label in small_area_labels:
        binary_image[labels == label] = 0

    # 定义结构元素（这里使用一个较小的矩形结构元素，你可以根据实际情况调整）
    kernel = np.ones((7, 7), np.uint8)

    # 执行膨胀操作，迭代两次（你可以根据需要调整结构元素的大小和迭代次数）
    dilate_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 使用 Canny 边缘检测算法
    edges = cv2.Canny(dilate_image, 0, 1)  # 调整参数以适应不同图像

    # 寻找边缘点的坐标
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig1, ax1 = plt.subplots(nrows=1, ncols=3)
    ax1[0].imshow(img)
    ax1[1].imshow(original)
    ax1[2].imshow(dilate_image)

    fig2, ax2 = plt.subplots()
    ax2.imshow(img)

    # 在图像上绘制多边形
    for i, contour in enumerate(contours):
        # 将轮廓坐标转换为 (x, y) 格式
        contour_points = contour.squeeze()

        # 寻找包围多边形的最小矩形
        rect = cv2.minAreaRect(contour_points)

        # 将最小矩形的信息转换为矩形的四个顶点坐标
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        poly = Polygon(box)
        distance = poly.area * 1 / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        try:
            boxes = expanded.squeeze(axis=0)
        except:
            continue
        coordinates = [[point[0], point[1]] for point in boxes]

        image = Image.open(input_pic)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        polygon = [tuple(row) for row in coordinates]
        draw.polygon(polygon, fill=255)

        # Combine the mask with the original image
        masked_image = Image.new("RGBA", image.size)
        masked_image.paste(image, (0, 0), mask)

        # Find the bounding box of the polygon
        bbox = masked_image.getbbox()

        # Crop the image using the bounding box
        cropped_image = masked_image.crop(bbox)

        # Generate the output file path
        output_dir = "C:\\Users\\14485\\Pictures\\temp_image"
        if not os.path.exists(f"{output_dir}\\{t}"):
            os.mkdir(f"{output_dir}\\{t}")
        output_path = f"{output_dir}\\{t}\\text_{i}.png"

        # Save the cropped image
        cropped_image.save(output_path)

        x, y = zip(*coordinates)

        # 闭合轮廓
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # 绘制轮廓
        ax1[0].plot(x, y, color='red')
        ax1[2].plot(x, y, color='red')

        ax2.plot(x, y, color='red')
    if not os.path.exists("C:\\Users\\14485\\Pictures\\temp_image\\with_showmap"):
        os.mkdir("C:\\Users\\14485\\Pictures\\temp_image\\with_showmap")
    fig1.savefig(f"C:\\Users\\14485\\Pictures\\temp_image\\with_showmap\\example_plot_{t}.png")
    plt.close(fig1)
    if not os.path.exists("C:\\Users\\14485\\Pictures\\temp_image\\without_showmap"):
        os.mkdir("C:\\Users\\14485\\Pictures\\temp_image\\without_showmap")
    fig2.savefig(f"C:\\Users\\14485\\Pictures\\temp_image\\without_showmap\\example_plot_{t}.png")
    plt.close(fig2)
