import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_contours(x, img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    image = x[0].cpu().detach().numpy()
    _, binary_image = cv2.threshold(image, 0.3, 1, cv2.THRESH_BINARY)
    binary_image = cv2.resize(binary_image, (w, h))
    binary_image = np.uint8(binary_image)

    # 定义结构元素（这里使用一个较小的矩形结构元素，你可以根据实际情况调整）
    kernel = np.ones((5, 5), np.uint8)

    # 执行膨胀操作，迭代两次（你可以根据需要调整结构元素的大小和迭代次数）
    dilate_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 使用 Canny 边缘检测算法
    edges = cv2.Canny(dilate_image, 0, 1)  # 调整参数以适应不同图像

    # 寻找边缘点的坐标
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     print("contours:{}".format(contours))
    #     plt.subplot(1,3,1), plt.imshow(img)
    #     plt.subplot(1,3,2), plt.imshow(binary_image)
    #     plt.subplot(1, 3, 3), plt.imshow(dilate_image)
    #     plt.show()
    return contours
