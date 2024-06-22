from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QMouseEvent, QPaintEvent, QLinearGradient, QColor, QPainter, QBrush, QPainterPath


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.m_dragPos = QPoint()
        self.setObjectName("MyWidget")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)

    def paintEvent(self, event: QPaintEvent):
        gradient = QLinearGradient(0, 0, 400, 0)  # 线性渐变的起始点和结束点的坐标
        gradient.setColorAt(0, QColor(31, 72, 132, 240))  # 添加颜色点
        gradient.setColorAt(1, QColor(214, 238, 245))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 在绘图时启用抗锯齿效果
        painter.setBrush(QBrush(gradient))  # 将这个画刷颜色设置到QPainter对象中
        painter.setPen(QColor(31, 72, 132, 230))  # 设置绘图时的画笔颜色

        cornerSize = 50  # 矩形大小
        arcR = cornerSize / 2  # 圆弧半径

        pathLeft = QPainterPath()  # 用于描述和绘制复杂路径的类
        rectLeft = QRect(0, 0, 401, 630)

        pathLeft.moveTo(rectLeft.left() + arcR, rectLeft.top())  # 移动到左上角
        pathLeft.arcTo(rectLeft.left(), rectLeft.top(), cornerSize, cornerSize, 90.0, 90.0)  # 向逆时针方向以半径为arcR，绘制90度的圆弧

        pathLeft.lineTo(rectLeft.left(), rectLeft.bottom() - arcR)  # 移动到左下角
        pathLeft.arcTo(rectLeft.left(), rectLeft.bottom() - cornerSize, cornerSize, cornerSize, 180.0,
                       90.0)  # 向逆时针方向以半径为arcR，绘制90度的圆弧

        pathLeft.lineTo(rectLeft.bottomRight())  # 移动到右下角

        pathLeft.lineTo(rectLeft.topRight())  # 移动到右上角

        painter.fillPath(pathLeft, painter.brush())  # 使用painter中画刷填充路径

        pathRight = QPainterPath()
        rectRight = QRect(400, 0, 771, 630)

        painter.setPen(QColor(238, 240, 247))

        pathRight.moveTo(rectRight.topLeft())
        pathRight.lineTo(rectRight.bottomLeft())

        pathRight.lineTo(rectRight.right() - arcR, rectRight.bottom())
        pathRight.arcTo(rectRight.right() - cornerSize, rectRight.bottom() - cornerSize, cornerSize, cornerSize, -90.0,
                        90.0)

        pathRight.lineTo(rectRight.right(), rectRight.top() - arcR)
        pathRight.arcTo(rectRight.right() - cornerSize, rectRight.top(), cornerSize, cornerSize, 0.0, 90.0)

        painter.fillPath(pathRight, QBrush(QColor(214, 238, 245)))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.m_dragPos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) and self.isTitleBar():
            self.move(event.globalPos() - self.m_dragPos)
            event.accept()

    def isTitleBar(self):
        return (0 < self.m_dragPos.x() < 1150) and (0 < self.m_dragPos.y() < 100)
