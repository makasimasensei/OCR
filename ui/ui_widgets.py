import glob
import os
import re
import shutil
import cv2
from PyQt5.QtCore import QRect, Qt, QUrl, QSize, QTimer, QPropertyAnimation, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QDesktopServices, QIcon
from PyQt5.QtWidgets import QWidget, QPushButton, QTabWidget, QLabel, QGridLayout, QLineEdit, QFileDialog, \
    QGraphicsOpacityEffect, QTextEdit, QStackedWidget, QHBoxLayout, QCheckBox
from qfluentwidgets.components.widgets import ComboBox
from PostProcess.postprint import post_print
from Recognition.Recognition import *
from database import *
from detection.test import Detection
from ui.MyLabel import MyLabel
from ui.MyScrollArea import MyScrollArea
from ui.MyTextEdit import MyTextEdit
from ui.MyWidget import MyWidget


class Ui_Widget(QWidget):
    def __init__(self):
        super().__init__()
        """初始化界面"""
        self.Widget = MyWidget()
        self.closeButton = QPushButton(self.Widget)
        self.minimizeButton = QPushButton(self.Widget)

        """左侧导航栏"""
        self.buttonStyle = (
            "QPushButton{background-color: #f9f6f9;border: 2px;border-style: outset;border-color: balck;border-radius: 9px;}\n"
            "QPushButton:hover{ background-color: #c9d4e3;}\n"
            "QPushButton:pressed {background-color: #8fb2e5;}"
        )
        self.pageButtonWidget = QWidget(self.Widget)
        self.hbox = QHBoxLayout(self.pageButtonWidget)
        self.pageButton_1 = QPushButton(self.pageButtonWidget)
        self.pageButton_2 = QPushButton(self.pageButtonWidget)
        self.pageButton_3 = QPushButton(self.pageButtonWidget)
        self.pageButton_4 = QPushButton(self.pageButtonWidget)
        self.tabWidget = QTabWidget(self.Widget)

        self.ui_1, self.ui_2, self.ui_3, self.ui_4 = QWidget(), QWidget(), QWidget(), QWidget()

        """Ui界面1"""
        self.ui_1_picLabel = QLabel(self.ui_1)
        self.ui_1_label_1 = QLabel(self.ui_1)
        self.ui_1_label_2 = QLabel(self.ui_1)
        self.ui_1_label_3 = QLabel(self.ui_1)
        self.ui_1_pushButton = QPushButton(self.ui_1)

        """Ui界面2"""
        self.ui_2_scrollArea = MyScrollArea(self.ui_2)
        self.ui_2_scrollLabel = MyLabel(self.ui_2)
        self.ui_2_scrollWidget = QWidget(self.ui_2_scrollArea)
        self.ui_2_scrollLayout = QGridLayout(self.ui_2_scrollWidget)
        self.ui_2_addButton = QPushButton(self.ui_2)
        self.ui_2_resetButton = QPushButton(self.ui_2)
        self.ui_2_label_1 = QLabel(self.ui_2)
        self.ui_2_textEdit = MyTextEdit(self.ui_2)
        self.ui_2_documentButton = QPushButton(self.ui_2)
        self.ui_2_label_2 = QLabel(self.ui_2)
        self.ui_2_comboBox_1 = ComboBox(self.ui_2)
        self.ui_2_comboBox_text_1 = "ResNet"
        self.ui_2_label_3 = QLabel(self.ui_2)
        self.ui_2_comboBox_2 = ComboBox(self.ui_2)
        self.ui_2_comboBox_text_2 = "without FPN"
        self.ui_2_confirmButton = QPushButton(self.ui_2)
        self.ui_2_loadLabel = QLabel(self.ui_2)
        self.ui_2_opacity_effect = QGraphicsOpacityEffect()
        self.ui_2_fadeAnimation = QPropertyAnimation(self.ui_2_opacity_effect, b"opacity")

        """Ui界面3"""
        self.ui_3_pageWidget = QStackedWidget(self.ui_3)
        self.ui_3_pushButton_left = QPushButton(self.ui_3)
        self.ui_3_pushButton_right = QPushButton(self.ui_3)
        self.ui_3_pageEdit = QLineEdit(self.ui_3)
        self.ui_3_totalPage = QLabel(self.ui_3)
        self.ui_3_checkBox = QCheckBox(self.ui_3)

        """Ui界面4"""
        self.ui_4_textEdit = QTextEdit(self.ui_4)
        self.ui_4_recButton = QPushButton(self.ui_4)
        self.ui_4_loadLabel = QLabel(self.ui_4)
        self.ui_4_opacity_effect = QGraphicsOpacityEffect()
        self.ui_4_fadeAnimation = QPropertyAnimation(self.ui_4_opacity_effect, b"opacity")

    """初始化界面"""

    def setupWidget(self):
        if self.Widget.objectName() == "":
            self.Widget.setObjectName("Widget")
        self.Widget.resize(1170, 630)
        self.Widget.show()

    def setupCloseButton(self):
        font = QFont()
        font.setPointSize(14)
        self.closeButton.setObjectName("closeButton")
        self.closeButton.setGeometry(QRect(1120, 0, 50, 30))
        self.closeButton.setFont(font)
        self.closeButton.setStyleSheet(
            "QPushButton{background-color:#d6eef5;border-top-right-radius: 25px;border-bottom-left-radius: 10px;}\n"
            "QPushButton:hover{ background-color: #bed7df;}")
        self.closeButton.setText("×")
        self.closeButton.clicked.connect(self.Widget.close)

    def setupMinimizeButton(self):
        font = QFont()
        font.setPointSize(14)
        self.minimizeButton.setObjectName("minimizeButton")
        self.minimizeButton.setGeometry(QRect(1070, 0, 50, 30))
        self.minimizeButton.setFont(font)
        self.minimizeButton.setStyleSheet(
            "QPushButton{background-color:#d6eef5;border-bottom-left-radius: 10px;border-bottom-right-radius: 10px}\n"
            "QPushButton:hover{ background-color: #bed7df;}")
        self.minimizeButton.setText("_")
        self.minimizeButton.clicked.connect(self.Widget.showMinimized)

    """左侧导航栏"""

    def setupPageButtonWidget(self):
        self.pageButtonWidget.setObjectName("pageButtonWidget")
        self.pageButtonWidget.setGeometry(QRect(0, 0, 100, 630))

    def setupQHBoxLayout(self):
        self.pageButtonWidget.setLayout(self.hbox)

    def setupPageButton_1(self):
        font = QFont()
        font.setFamily("KaiTi")
        font.setPointSize(12)
        self.pageButton_1.setObjectName("pageButton_1")
        self.pageButton_1.setGeometry(QRect(10, 50, 90, 50))
        self.pageButton_1.setFont(font)
        self.pageButton_1.setStyleSheet(self.buttonStyle)
        self.pageButton_1.setText("简介")
        self.pageButton_1.clicked.connect(self.pageButtonClick_1)

    def setupPageButton_2(self):
        font = QFont()
        font.setFamily("KaiTi")
        font.setPointSize(12)
        self.pageButton_2.setObjectName("pageButton_2")
        self.pageButton_2.setGeometry(QRect(10, 80, 90, 50))
        self.pageButton_2.setFont(font)
        self.pageButton_2.setStyleSheet(self.buttonStyle)
        self.pageButton_2.setText("参数设置")
        self.pageButton_2.clicked.connect(self.pageButtonClick_2)

    def setupPageButton_3(self):
        font = QFont()
        font.setFamily("KaiTi")
        font.setPointSize(12)
        self.pageButton_3.setObjectName("pageButton_3")
        self.pageButton_3.setGeometry(QRect(10, 110, 90, 50))
        self.pageButton_3.setFont(font)
        self.pageButton_3.setStyleSheet(self.buttonStyle)
        self.pageButton_3.setText("检测结果")
        self.pageButton_3.clicked.connect(self.pageButtonClick_3)

    def setupPageButton_4(self):
        font = QFont()
        font.setFamily("KaiTi")
        font.setPointSize(12)
        self.pageButton_4.setObjectName("pageButton_4")
        self.pageButton_4.setGeometry(QRect(10, 140, 90, 50))
        self.pageButton_4.setFont(font)
        self.pageButton_4.setStyleSheet(self.buttonStyle)
        self.pageButton_4.setText("识别结果")
        self.pageButton_4.clicked.connect(self.pageButtonClick_4)
        self.pageButtonRaise(1)

    def setupTabWidget(self):
        self.tabWidget.setTabPosition(QTabWidget.West)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setStyleSheet(
            "QTabBar::tab{width: 0;height: 0;margin: 0;padding: 0;border: none;}\n"
            "QTabWidget::pane {border: 3px solid #ff4242;border-radius: 12px;border-style: outset;}"
        )
        self.tabWidget.setGeometry(QRect(90, 40, 1050, 560))
        self.tabWidget.addTab(self.ui_1, "1")
        self.tabWidget.addTab(self.ui_2, "2")
        self.tabWidget.addTab(self.ui_3, "3")
        self.tabWidget.addTab(self.ui_4, "4")

    """Ui界面1"""

    def setupUi_1(self):
        self.ui_1.setStyleSheet(
            "QWidget{background-color:#e4f2f7;border: 2px;border-color: #000000;border-radius: 12px;}")

    def setupUi1PicLabel(self):
        self.ui_1_picLabel.setObjectName("ui_1_picLabel")
        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\rounded_image.png")
        self.ui_1_picLabel.setPixmap(pixmap)
        self.ui_1_picLabel.setGeometry(0, 0, 560, 560)
        self.ui_1_picLabel.resize(pixmap.size())
        self.ui_1_picLabel.show()

    def setupUi1Label_1(self):
        self.ui_1_label_1.setGeometry(QRect(605, 165, 400, 30))
        ui_1_font1 = QFont()
        ui_1_font1.setFamily("KaiTi")
        ui_1_font1.setPointSize(26)
        ui_1_font1.setBold(True)
        self.ui_1_label_1.setFont(ui_1_font1)
        self.ui_1_label_1.setAlignment(Qt.AlignCenter)
        self.ui_1_label_1.setObjectName("ui_1_label_1")
        self.ui_1_label_1.setText("场景文本检测与识别软件")

    def setupUi1Label_2(self):
        self.ui_1_label_2.setGeometry(QRect(605, 225, 400, 30))
        ui_1_font1 = QFont()
        ui_1_font1.setFamily("KaiTi")
        ui_1_font1.setPointSize(24)
        ui_1_font1.setBold(True)
        self.ui_1_label_2.setFont(ui_1_font1)
        self.ui_1_label_2.setAlignment(Qt.AlignCenter)
        self.ui_1_label_2.setObjectName("ui_1_label_1")
        self.ui_1_label_2.setText("V2.0")

    def setupUi1Label_3(self):
        self.ui_1_label_3.setGeometry(QRect(605, 285, 400, 30))
        ui_1_font2 = QFont()
        ui_1_font2.setFamily("KaiTi")
        ui_1_font2.setPointSize(16)
        self.ui_1_label_3.setFont(ui_1_font2)
        self.ui_1_label_3.setAlignment(Qt.AlignCenter)
        self.ui_1_label_3.setObjectName("ui_1_label_1")
        self.ui_1_label_3.setText("软件详情请查看使用说明")

    def setupUi1PushButton(self):
        self.ui_1_pushButton.setGeometry(QRect(745, 355, 120, 40))
        ui_1_font2 = QFont()
        ui_1_font2.setFamily("KaiTi")
        ui_1_font2.setPointSize(16)
        self.ui_1_pushButton.setFont(ui_1_font2)
        ui_1_pageButtonstyle = (
            "QPushButton{color: white;background-color: #2658a1;border:none;border-style: outset;border-color: #737373;border-radius: 14px;}"
            "QPushButton:hover{ background-color: #2f6cc6;}"
            "QPushButton:pressed {background-color: #224d8c;}"
        )
        self.ui_1_pushButton.setStyleSheet(ui_1_pageButtonstyle)
        self.ui_1_pushButton.setIcon(QIcon("E:\\PyCharm\\OCR\\image\\net.png"))
        self.ui_1_pushButton.setIconSize(QSize(15, 15))
        self.ui_1_pushButton.setText("使用说明")

    """Ui界面2"""

    def setupUi_2(self):
        self.ui_2.setStyleSheet(
            "QWidget{background-color:#e4f2f7;border: 2px;border-color: #000000;border-radius: 12px;}")

    def setupUi2ScrollArea(self):
        self.ui_2_scrollArea.setGeometry(QRect(10, 10, 500, 500))
        self.ui_2_scrollArea.setStyleSheet(
            "ScrollArea{background-color:#ededed;border: 2px dashed black;border-color: #000000;padding: 5px;border-radius: 12px;}")
        self.ui_2_scrollArea.setObjectName("ui_2_scrollArea")
        self.ui_2_scrollArea.setWidgetResizable(True)
        self.ui_2_scrollArea.fileDropped.connect(self.handleFileDropped)

    def setupUi2ScrollLabel(self):
        parentWidth = self.ui_2_scrollArea.width()
        parentHeight = self.ui_2_scrollArea.height()
        childWidth = 200
        childHeight = 200
        childX = int((parentWidth - childWidth) / 2)
        childY = int((parentHeight - childHeight) / 2)
        self.ui_2_scrollLabel.setGeometry(childX, childY, childWidth, childHeight)
        palette = self.ui_2_scrollLabel.palette()
        palette.setColor(QPalette.WindowText, QColor("#888888"))
        self.ui_2_scrollLabel.setPalette(palette)
        scrollLabelfont = QFont()
        scrollLabelfont.setFamily("KaiTi")
        scrollLabelfont.setPointSize(14)
        self.ui_2_scrollLabel.setFont(scrollLabelfont)
        self.ui_2_scrollLabel.setAlignment(Qt.AlignCenter)
        self.ui_2_scrollLabel.setText("拖拽图片到此处")
        self.ui_2_scrollLabel.fileDropped.connect(self.handleFileDropped)

    def setupUi2ScrollWidget(self):
        self.ui_2_scrollWidget.setLayout(self.ui_2_scrollLayout)
        self.ui_2_scrollArea.setWidget(self.ui_2_scrollWidget)

    def setupUi2AddButton(self):
        self.ui_2_addButton.setGeometry(QRect(10, 515, 30, 30))
        self.ui_2_addButton.setStyleSheet(
            "QPushButton{background-color:transparent;border: 1px transparent;border-radius: 9px;}"
            "QPushButton:hover {background-color: #bbdce8;}"
            "QPushButton:pressed {background-color: #93c5d7;}")
        self.ui_2_addButton.setObjectName("ui_2_resetButton")
        self.ui_2_addButton.setIcon(QIcon("E:\\PyCharm\\OCR\\image\\add.png"))
        self.ui_2_addButton.setIconSize(QSize(23, 23))
        self.ui_2_addButton.clicked.connect(self.addClicked)

    def setupUi2ResetButton(self):
        self.ui_2_resetButton.setGeometry(QRect(470, 515, 30, 30))
        self.ui_2_resetButton.setStyleSheet(
            "QPushButton{background-color:transparent;border: 1px transparent;border-radius: 9px;}"
            "QPushButton:hover {background-color: #bbdce8;}"
            "QPushButton:pressed {background-color: #93c5d7;}")
        self.ui_2_resetButton.setObjectName("ui_2_resetButton")
        self.ui_2_resetButton.setIcon(QIcon("E:\\PyCharm\\OCR\\image\\reset.png"))
        self.ui_2_resetButton.setIconSize(QSize(20, 20))
        self.ui_2_resetButton.clicked.connect(self.resetClicked)

    def setupUi2Label_1(self):
        self.ui_2_label_1.setGeometry(QRect(570, 40, 100, 15))
        ui_2_font = QFont()
        ui_2_font.setFamily("KaiTi")
        ui_2_font.setPointSize(12)
        self.ui_2_label_1.setFont(ui_2_font)
        self.ui_2_label_1.setObjectName("ui_2_label_1")
        self.ui_2_label_1.setText("文件路径")

    def setupUi2TextEdit(self):
        self.ui_2_textEdit.setGeometry(QRect(570, 60, 400, 260))
        ui_2_text_font = QFont()
        ui_2_text_font.setFamily("Times New Roman")
        ui_2_text_font.setPointSize(14)
        self.ui_2_textEdit.setFont(ui_2_text_font)
        self.ui_2_textEdit.setStyleSheet(
            "QTextEdit{background-color:#ffffff;border: 1px solid #ffffff;border-style: outset;border-radius: 9px;}"
            "QTextEdit:hover {background-color: #ebebeb;}")
        self.ui_2_textEdit.setObjectName("ui_2_textEdit")
        self.ui_2_textEdit.setPlaceholderText("C:/Users/14485/Pictures/Screenshots/gt_0.jpg")
        self.ui_2_textEdit.returnPressed.connect(self.handleReturnPressed)
        self.ui_2_textEdit.fileDropped.connect(self.handleFileDropped)

    def setupUi2DocumentButton(self):
        self.ui_2_documentButton.setGeometry(QRect(990, 280, 40, 40))
        self.ui_2_documentButton.setStyleSheet(
            "QPushButton{background-color:#ffffff;border: 1px #ffffff;border-style: outset;border-radius: 9px;}"
            "QPushButton:hover {background-color: #e0e0e0;}"
            "QPushButton:pressed {background-color: #d6d6d6;}")
        self.ui_2_documentButton.setObjectName("ui_2_documentButton")
        self.ui_2_documentButton.setIcon(QIcon("E:\\PyCharm\\OCR\\image\\document.png"))
        self.ui_2_documentButton.setIconSize(QSize(30, 30))
        self.ui_2_documentButton.clicked.connect(self.documentClicked)

    def setupUi2Label_2(self):
        self.ui_2_label_2.setGeometry(QRect(570, 330, 100, 15))
        ui_2_font = QFont()
        ui_2_font.setFamily("KaiTi")
        ui_2_font.setPointSize(12)
        self.ui_2_label_2.setFont(ui_2_font)
        self.ui_2_label_2.setObjectName("ui_2_label_2")
        self.ui_2_label_2.setText("骨干网络")

    def setupUi2ComboBox_1(self):
        self.ui_2_comboBox_1.setGeometry(QRect(570, 350, 210, 40))
        self.ui_2_comboBox_1.setAcceptDrops(False)
        self.ui_2_comboBox_1.setObjectName("ui_2_comboBox_1")
        self.ui_2_comboBox_1.setStyleSheet("ComboBox {\n"
                                           "    border: 1px solid #ffffff;\n"
                                           "    border-radius: 9px;\n"
                                           "    padding: 5px 31px 6px 11px;\n"
                                           "    font: 16px \'Times New Roman\';\n"
                                           "    color: black;\n"
                                           "    background-color: #ffffff;\n"
                                           "    text-align: left;\n"
                                           "    outline: none;\n"
                                           "}\n"
                                           "ComboBox:hover {\n"
                                           "    background-color: #e0e0e0;\n"
                                           "}\n"
                                           "ComboBox:pressed {\n"
                                           "    background-color: rgba(249, 249, 249, 0.3);\n"
                                           "    color: #d6d6d6;\n"
                                           "}\n")
        self.ui_2_comboBox_1.addItem("ResNet")
        self.ui_2_comboBox_1.addItem("MobileNet")
        self.ui_2_comboBox_1.addItem("ESNet")
        self.ui_2_comboBox_1.currentIndexChanged.connect(self.comboBox1SelectionChanged)

    def setupUi2Label_3(self):
        self.ui_2_label_3.setGeometry(QRect(570, 400, 100, 15))
        ui_2_font = QFont()
        ui_2_font.setFamily("KaiTi")
        ui_2_font.setPointSize(12)
        self.ui_2_label_3.setFont(ui_2_font)
        self.ui_2_label_3.setObjectName("ui_2_label_3")
        self.ui_2_label_3.setText("特征融合网络")

    def setupUi2ComboBox_2(self):
        self.ui_2_comboBox_2.setGeometry(QRect(570, 420, 210, 40))
        self.ui_2_comboBox_2.setAcceptDrops(False)
        self.ui_2_comboBox_2.setObjectName("ui_2_comboBox_2")
        self.ui_2_comboBox_2.setStyleSheet("ComboBox {\n"
                                           "    border: 1px solid #ffffff;\n"
                                           "    border-radius: 9px;\n"
                                           "    padding: 5px 31px 6px 11px;\n"
                                           "    font: 16px \'Times New Roman\';\n"
                                           "    color: black;\n"
                                           "    background-color: #ffffff;\n"
                                           "    text-align: left;\n"
                                           "    outline: none;\n"
                                           "}\n"
                                           "ComboBox:hover {\n"
                                           "    background-color: #e0e0e0;\n"
                                           "}\n"
                                           "ComboBox:pressed {\n"
                                           "    background-color: rgba(249, 249, 249, 0.3);\n"
                                           "    color: #d6d6d6;\n"
                                           "}\n")
        self.ui_2_comboBox_2.addItem("without FPN")
        self.ui_2_comboBox_2.addItem("FPN")
        self.ui_2_comboBox_2.addItem("WBFFPN")
        self.ui_2_comboBox_2.currentIndexChanged.connect(self.comboBox2SelectionChanged)

    def setupUi2ConfirmButton(self):
        self.ui_2_confirmButton.setGeometry(QRect(570, 490, 110, 40))
        ui_2_button_font = QFont()
        ui_2_button_font.setFamily("楷体")
        ui_2_button_font.setPointSize(20)
        ui_2_button_font.setWeight(50)
        self.ui_2_confirmButton.setFont(ui_2_button_font)
        self.ui_2_confirmButton.setStyleSheet(
            "QPushButton{background-color:#ffffff;border: 1px #ffffff;border-style: outset;border-radius: 9px;}"
            "QPushButton:hover {background-color: #e0e0e0;}"
            "QPushButton:pressed {background-color: #d6d6d6;}")
        self.ui_2_confirmButton.setObjectName("pushButton")
        self.ui_2_confirmButton.setText("确认")
        self.ui_2_confirmButton.clicked.connect(self.ui2ConfirmClicked)

    def setupUi2LoadLabel(self):
        self.ui_2_loadLabel.setGeometry(QRect(700, 495, 30, 30))
        self.ui_2_loadLabel.setObjectName("ui_2_loadLabel")
        self.ui_2_loadLabel.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\load_image.png")
        self.ui_2_loadLabel.setPixmap(pixmap)
        self.ui_2_opacity_effect.setOpacity(0)
        self.ui_2_loadLabel.setGraphicsEffect(self.ui_2_opacity_effect)

    """Ui界面3"""

    def setupUi_3(self):
        self.ui_3.setStyleSheet(
            "QWidget{background-color:#e4f2f7;border: 2px;border-color: #000000;border-radius: 12px;}")

    def setupPageWidget(self):
        self.ui_3_pageWidget.setGeometry(QRect(10, 10, 1025, 500))
        self.ui_3_pageWidget.setStyleSheet(
            "QStackedWidget {border: 2px dashed black;background-color: #ffffff;}"
        )

    def setupUi3PushButtonLeft(self):
        self.ui_3_pushButton_left.setGeometry(QRect(400, 510, 40, 40))
        self.ui_3_pushButton_left.setObjectName("ui_3_label_left")
        self.ui_3_pushButton_left.setStyleSheet(
            "QPushButton{background-color:#ffffff;border: 1px #ffffff;border-style: outset;border-radius: 9px;}"
            "QPushButton:hover {background-color: #e0e0e0;}"
            "QPushButton:pressed {background-color: #d6d6d6;}")
        self.ui_3_pushButton_left.setIcon(QIcon("E:\\PyCharm\\OCR\\image\\上页.png"))
        self.ui_3_pushButton_left.setIconSize(QSize(30, 30))
        self.ui_3_pushButton_left.clicked.connect(self.ui3ShowPrevPage)

    def setupUi3PushButtonRight(self):
        self.ui_3_pushButton_right.setGeometry(QRect(600, 510, 40, 40))
        self.ui_3_pushButton_right.setObjectName("ui_3_label_left")
        self.ui_3_pushButton_right.setStyleSheet(
            "QPushButton{background-color:#ffffff;border: 1px #ffffff;border-style: outset;border-radius: 9px;}"
            "QPushButton:hover {background-color: #e0e0e0;}"
            "QPushButton:pressed {background-color: #d6d6d6;}")
        self.ui_3_pushButton_right.setIcon(QIcon("E:\\PyCharm\\OCR\\image\\下页.png"))
        self.ui_3_pushButton_right.setIconSize(QSize(30, 30))
        self.ui_3_pushButton_right.clicked.connect(self.ui3ShowNextPage)

    def setupUi3PageEdit(self):
        self.ui_3_pageEdit.setGeometry(QRect(485, 510, 30, 40))
        ui_3_line_font = QFont()
        ui_3_line_font.setFamily("Times New Roman")
        ui_3_line_font.setPointSize(16)
        self.ui_3_pageEdit.setFont(ui_3_line_font)
        self.ui_3_pageEdit.setAlignment(Qt.AlignCenter)
        self.ui_3_pageEdit.setStyleSheet(
            "QLineEdit{background-color:#ffffff;border: 1px solid #ffffff;border-style: outset;border-radius: 0px;}"
            "QLineEdit:hover {background-color: #ebebeb;}")
        self.ui_3_pageEdit.setObjectName("ui_3_pageEdit")
        self.ui_3_pageEdit.setPlaceholderText("0")
        self.ui_3_pageEdit.returnPressed.connect(self.ui3ToPage)

    def setupUi3TotalPage(self):
        self.ui_3_totalPage.setGeometry(QRect(515, 510, 30, 40))
        ui_3_line_font = QFont()
        ui_3_line_font.setFamily("Times New Roman")
        ui_3_line_font.setPointSize(16)
        self.ui_3_totalPage.setStyleSheet(
            "QLabel{background-color:#ffffff;border: 1px solid #ffffff;border-style: outset;border-radius: 0px;}")
        self.ui_3_totalPage.setFont(ui_3_line_font)
        self.ui_3_totalPage.setAlignment(Qt.AlignCenter)
        self.ui_3_totalPage.setObjectName("ui_3_totalPage")
        self.ui_3_totalPage.setText("/0")

    def setupUi3ChenckBox(self):
        self.ui_3_checkBox.setObjectName("ui_3_label")
        self.ui_3_checkBox.setGeometry(QRect(940, 515, 100, 30))
        ui_3_label_font = QFont()
        ui_3_label_font.setFamily("KaiTi")
        ui_3_label_font.setPointSize(14)
        self.ui_3_checkBox.setFont(ui_3_label_font)
        self.ui_3_checkBox.setText("阈值图")
        self.ui_3_checkBox.stateChanged.connect(self.ui3CheckboxStateChanged)

    """Ui界面4"""

    def setupUi_4(self):
        self.ui_4.setStyleSheet(
            "QWidget{background-color:#e4f2f7;border: 2px;border-color: #000000;border-radius: 12px;}")

    def ui4TextEdit(self):
        self.ui_4_textEdit.setGeometry(QRect(10, 10, 1025, 500))
        self.ui_4_textEdit.setStyleSheet(
            "QTextEdit{background-color:#ffffff;border: 2px dashed black;border-color: #000000;padding: 5px;border-radius: 12px;}")
        self.ui_4_textEdit.setObjectName("ui_4_textEdit")
        ui_4_text_font = QFont()
        ui_4_text_font.setFamily("KaiTi")
        ui_4_text_font.setPointSize(14)
        self.ui_4_textEdit.setFont(ui_4_text_font)

    def ui4RecButton(self):
        self.ui_4_recButton.setGeometry(QRect(850, 510, 120, 40))
        ui_4_button_font = QFont()
        ui_4_button_font.setFamily("KaiTi")
        ui_4_button_font.setPointSize(20)
        ui_4_button_font.setWeight(50)
        self.ui_4_recButton.setFont(ui_4_button_font)
        self.ui_4_recButton.setText("识别图片")
        self.ui_4_recButton.setStyleSheet(
            "QPushButton{background-color:#ffffff;border: 1px #ffffff;border-style: outset;border-radius: 9px;}"
            "QPushButton:hover {background-color: #e0e0e0;}"
            "QPushButton:pressed {background-color: #d6d6d6;}")
        self.ui_4_recButton.setObjectName("ui_4_recButton")
        self.ui_4_recButton.clicked.connect(self.ui4ConfirmClicked)

    def setupUi4LoadLabel(self):
        self.ui_4_loadLabel.setGeometry(QRect(980, 518, 30, 30))
        self.ui_4_loadLabel.setObjectName("ui_4_loadLabel")
        self.ui_4_loadLabel.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\load_image.png")
        self.ui_4_loadLabel.setPixmap(pixmap)
        self.ui_4_opacity_effect.setOpacity(0)
        self.ui_4_loadLabel.setGraphicsEffect(self.ui_4_opacity_effect)

    """槽函数"""

    def pageButtonClick_1(self):
        self.pageButtonRaise(1)
        self.tabWidget.setCurrentIndex(0)

    def pageButtonClick_2(self):
        self.pageButtonRaise(2)
        self.tabWidget.setCurrentIndex(1)

    def pageButtonClick_3(self):
        self.pageButtonRaise(3)
        self.tabWidget.setCurrentIndex(2)

    def pageButtonClick_4(self):
        self.pageButtonRaise(4)
        self.tabWidget.setCurrentIndex(3)

    def pageButtonRaise(self, target):
        bs_red = (
            "QPushButton{background-color: #f9f6f9;border: 3px;border-style: outset;border-color: red;border-radius: 9px;}"
            "QPushButton:hover{ background-color: #c9d4e3;}"
            "QPushButton:pressed {background-color: #8fb2e5;}"
        )
        bs_black = (
            "QPushButton{background-color: #f9f6f9;border: 2px;border-style: outset;border-color: black;border-radius: 9px;}"
            "QPushButton:hover{ background-color: #c9d4e3;}"
            "QPushButton:pressed {background-color: #8fb2e5;}"
        )
        qr_big = QRect(0, 20 + 30 * target, 100, 50)
        font = QFont()
        font.setFamily("KaiTi")
        font.setPointSize(14)
        font.setBold(True)

        if target == 1:
            self.pageButton_1.setGeometry(qr_big)
            self.pageButton_1.setStyleSheet(bs_red)
            self.pageButton_1.setFont(font)
            self.pageButton_1.lower()
        if target == 2:
            self.pageButton_2.setGeometry(qr_big)
            self.pageButton_2.setStyleSheet(bs_red)
            self.pageButton_2.setFont(font)
            self.pageButton_2.lower()
        if target == 3:
            self.pageButton_3.setGeometry(qr_big)
            self.pageButton_3.setStyleSheet(bs_red)
            self.pageButton_3.setFont(font)
            self.pageButton_3.lower()
        if target == 4:
            self.pageButton_4.setGeometry(qr_big)
            self.pageButton_4.setStyleSheet(bs_red)
            self.pageButton_4.setFont(font)
            self.pageButton_4.lower()

        numbers = [1, 2, 3, 4]
        distances = [(num, abs(num - target)) for num in numbers]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        sorted_numbers = [num for num, _ in sorted_distances]
        sorted_numbers.remove(target)

        font.setPointSize(12)
        font.setBold(False)
        for l in sorted_numbers:
            if l == 1:
                self.pageButton_1.setGeometry(QRect(10, 50, 90, 50))
                self.pageButton_1.setStyleSheet(bs_black)
                self.pageButton_1.setFont(font)
                self.pageButton_1.lower()
            if l == 2:
                self.pageButton_2.setGeometry(QRect(10, 80, 90, 50))
                self.pageButton_2.setStyleSheet(bs_black)
                self.pageButton_2.setFont(font)
                self.pageButton_2.lower()
            if l == 3:
                self.pageButton_3.setGeometry(QRect(10, 110, 90, 50))
                self.pageButton_3.setStyleSheet(bs_black)
                self.pageButton_3.setFont(font)
                self.pageButton_3.lower()
            if l == 4:
                self.pageButton_4.setGeometry(QRect(10, 140, 90, 50))
                self.pageButton_4.setStyleSheet(bs_black)
                self.pageButton_4.setFont(font)
                self.pageButton_4.lower()

    @staticmethod
    def openUrl():
        url = QUrl("https://pan.baidu.com/s/1DfO9-ip1t-QmPZ2CSZvwxQ?pwd=sh79")
        QDesktopServices.openUrl(url)

    def handleFileDropped(self, stringList):
        file_paths = ""
        currentPath = os.getcwd()

        children = self.ui_2_scrollArea.findChildren(QLabel)
        for child in children:
            child.deleteLater()

        self.ui_2_scrollLabel.setVisible(False)

        for a in range(1, len(stringList) + 1):
            label = QLabel("Label{}".format(a), self.ui_2_scrollArea)
            label.setStyleSheet("QLabel{border: None;}")
            label.setAlignment(Qt.AlignCenter)

            row = int((a - 1) / 4)
            cloumn = int((a - 1) % 4)
            self.ui_2_scrollLayout.addWidget(label, row, cloumn)

            if not file_paths:
                file_paths = stringList[a - 1]
            else:
                file_paths = file_paths + '\n' + stringList[a - 1]
            img = cv2.imread(stringList[a - 1])
            if len(stringList) == 1:
                img = cv2.resize(img, (400, 400))
            elif len(stringList) == 2:
                img = cv2.resize(img, (200, 200))
            elif len(stringList) >= 3:
                img = cv2.resize(img, (100, 100))
            path = currentPath + "\\image\\temp" + str(a) + ".jpg"
            cv2.imwrite(path, img)
            pixmap = QPixmap(path)
            label.setPixmap(pixmap)
            label.show()
            os.remove(path)
        self.ui_2_textEdit.setPlainText(file_paths)

    def handleReturnPressed(self):
        text = self.ui_2_textEdit.toPlainText()
        if not text:
            self.ui_2_textEdit.setPlainText("C:/Users/14485/Pictures/Screenshots/gt_0.jpg")
            stringList = ["C:/Users/14485/Pictures/Screenshots/gt_0.jpg"]
            self.handleFileDropped(stringList)
        else:
            text = self.ui_2_textEdit.toPlainText()
            stringList = text.split('\n')
            stringList = list(filter(None, stringList))
            self.handleFileDropped(stringList)

    def resetClicked(self):
        children = self.ui_2_scrollArea.findChildren(QLabel)
        for child in children:
            child.deleteLater()
        self.ui_2_scrollLabel.setVisible(True)
        self.ui_2_textEdit.setPlainText("")

    def addClicked(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "All Files (*)")
        if file_paths:
            string = self.ui_2_textEdit.toPlainText()
            if string:
                stringList = string.split('\n')
            else:
                stringList = []
            for file_path in file_paths:
                if not string:
                    string = file_path
                    stringList.append(file_path)
                else:
                    string = string + '\n' + file_path
                    stringList.append(file_path)
            self.ui_2_textEdit.setText(string)
            self.handleFileDropped(stringList)

    def documentClicked(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "All Files (*)")
        if file_paths:
            string = ""
            stringList = []
            for file_path in file_paths:
                if not string:
                    string = file_path
                    stringList.append(file_path)
                else:
                    string = string + '\n' + file_path
                    stringList.append(file_path)
            self.ui_2_textEdit.setText(string)
            self.handleFileDropped(stringList)

    def comboBox1SelectionChanged(self):
        self.ui_2_comboBox_text_1 = self.ui_2_comboBox_1.currentText()

    def comboBox2SelectionChanged(self):
        self.ui_2_comboBox_text_2 = self.ui_2_comboBox_2.currentText()

    def ui2ConfirmClicked(self):
        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\load_image.png")
        self.ui_2_loadLabel.setPixmap(pixmap)
        self.ui_2_opacity_effect.setOpacity(1)
        self.ui_2_loadLabel.setGraphicsEffect(self.ui_2_opacity_effect)

        QTimer.singleShot(300, self.detection_progress)

    def detection_progress(self):
        directory_path = "C:\\Users\\14485\\Pictures\\temp_image"
        files = glob.glob(os.path.join(directory_path, "*"))
        for file in files:
            try:
                shutil.rmtree(file)
            except Exception as e:
                print(f"Failed to delete {file}. Reason: {e}")

        text = self.ui_2_textEdit.toPlainText()
        text_list = text.split('\n')
        for t in range(len(text_list)):
            det = Detection(self.ui_2_comboBox_text_1, self.ui_2_comboBox_text_2)
            out = det(text_list[t])
            post_print(out, text_list[t], t)

        self.ui_3_pageEdit.setText('1')

        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\succeed.png")
        self.ui_2_loadLabel.setPixmap(pixmap)
        self.ui_2_loadLabel.setGraphicsEffect(self.ui_2_opacity_effect)

        # 创建属性动画
        self.ui_2_fadeAnimation.setDuration(2000)  # 动画持续时间（毫秒）
        self.ui_2_fadeAnimation.setStartValue(1)  # 开始透明度
        self.ui_2_fadeAnimation.setEndValue(0)  # 结束透明度
        self.ui_2_fadeAnimation.start()
        if self.ui_3_checkBox.isChecked():
            self.ui3ShowMorePic()
        else:
            self.ui3ShowPic()

    def ui3CheckboxStateChanged(self, state):
        if state == 0:
            self.ui3ShowPic()
        elif state == 2:
            self.ui3ShowMorePic()

    def ui3ShowMorePic(self):
        count = self.ui_3_pageWidget.count()
        if count != -1:
            for i in range(count - 1, -1, -1):
                widget_to_remove = self.ui_3_pageWidget.widget(i)
                self.ui_3_pageWidget.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()
        folder_path = "C:\\Users\\14485\\Pictures\\temp_image\\with_showmap"
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.lower().endswith('.png'):
                ui_3_detLabel = QLabel(self.ui_3)
                ui_3_detLabel.setObjectName("ui_3_detLabel")
                ui_3_detLabel.setGeometry(QRect(10, 10, 1025, 500))
                ui_3_detLabel.setStyleSheet("QLabel{background-color:#ffffff;border: none;border-radius: 9px;}")
                ui_3_detLabel.setAlignment(Qt.AlignCenter)
                self.ui_3_pageWidget.addWidget(ui_3_detLabel)
                pixmap = QPixmap(full_path)
                ui_3_detLabel.setPixmap(pixmap)
        self.ui_3_totalPage.setText('/' + str(self.ui_3_pageWidget.count()))

    def ui3ShowPic(self):
        count = self.ui_3_pageWidget.count()
        if count != 0:
            for i in range(count - 1, -1, -1):
                widget_to_remove = self.ui_3_pageWidget.widget(i)
                self.ui_3_pageWidget.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()
        folder_path = "C:\\Users\\14485\\Pictures\\temp_image\\without_showmap"
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.lower().endswith('.png'):
                ui_3_detLabel = QLabel(self.ui_3)
                ui_3_detLabel.setObjectName("ui_3_detLabel")
                ui_3_detLabel.setGeometry(QRect(10, 10, 1025, 500))
                ui_3_detLabel.setStyleSheet("QLabel{background-color:#ffffff;border: none;border-radius: 9px;}")
                ui_3_detLabel.setAlignment(Qt.AlignCenter)
                self.ui_3_pageWidget.addWidget(ui_3_detLabel)
                pixmap = QPixmap(full_path)
                ui_3_detLabel.setPixmap(pixmap)
        self.ui_3_totalPage.setText('/' + str(self.ui_3_pageWidget.count()))

    def ui3ShowPrevPage(self):
        current_index = self.ui_3_pageWidget.currentIndex()
        index = (current_index - 1) % self.ui_3_pageWidget.count() + 1
        self.ui_3_pageEdit.setText(str(index))
        self.ui_3_pageWidget.setCurrentIndex(index - 1)

    def ui3ShowNextPage(self):
        current_index = self.ui_3_pageWidget.currentIndex()
        index = (current_index + 1) % self.ui_3_pageWidget.count() + 1
        self.ui_3_pageEdit.setText(str(index))
        self.ui_3_pageWidget.setCurrentIndex(index - 1)

    def ui3ToPage(self):
        index = self.ui_3_pageEdit.text()
        self.ui_3_pageWidget.setCurrentIndex(int(index) - 1)

    def ui4ConfirmClicked(self):
        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\load_image.png")
        self.ui_4_loadLabel.setPixmap(pixmap)
        self.ui_2_opacity_effect.setOpacity(1)
        self.ui_4_loadLabel.setGraphicsEffect(self.ui_2_opacity_effect)

        QTimer.singleShot(300, self.ui4Recognition)

    def ui4Recognition(self):
        sb = Recognition()
        base_path = "C:\\Users\\14485\\Pictures\\temp_image"
        for entry in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, entry)) and re.match(r'^\d+$', entry):
                dir = os.path.join(base_path, entry)
                words = ""
                for filename in os.listdir(dir):
                    pic_path = os.path.join(dir, filename)
                    if filename.lower().endswith('.png') and sb.process_image(pic_path):
                        str(sb.process_image(pic_path))
                        if not words:
                            words = str(sb.process_image(pic_path))
                        else:
                            words = words + '\t' + str(sb.process_image(pic_path))
                self.ui_4_textEdit.append(words)
                self.saveIntoDatabase(int(entry), words)

        pixmap = QPixmap("E:\\PyCharm\\OCR\\image\\succeed.png")
        self.ui_4_loadLabel.setPixmap(pixmap)
        self.ui_4_loadLabel.setGraphicsEffect(self.ui_2_opacity_effect)

        # 创建属性动画
        self.ui_4_fadeAnimation.setDuration(2000)  # 动画持续时间（毫秒）
        self.ui_4_fadeAnimation.setStartValue(1)  # 开始透明度
        self.ui_4_fadeAnimation.setEndValue(0)  # 结束透明度
        self.ui_4_fadeAnimation.start()

    @staticmethod
    def saveIntoDatabase(num, words):
        path = "C:\\Users\\14485\\Pictures\\temp_image\\with_showmap"
        path = os.path.join(path, "example_plot_{}.png".format(num))
        with open(path, "rb") as file:
            binary_data = file.read()
            insert_into_database(words, binary_data)

    def setupUi(self):
        """构建UI"""

        """初始化界面"""
        self.setupWidget()
        self.setupCloseButton()
        self.setupMinimizeButton()

        """左侧导航栏"""
        self.setupPageButtonWidget()
        self.setupQHBoxLayout()
        self.setupPageButton_1()
        self.setupPageButton_2()
        self.setupPageButton_3()
        self.setupPageButton_4()
        self.setupTabWidget()

        """Ui界面1"""
        self.setupUi_1()
        self.setupUi1PicLabel()
        self.setupUi1Label_1()
        self.setupUi1Label_2()
        self.setupUi1Label_3()
        self.setupUi1PushButton()

        """Ui界面2"""
        self.setupUi_2()
        self.setupUi2ScrollArea()
        self.setupUi2ScrollLabel()
        self.setupUi2ScrollWidget()
        self.setupUi2AddButton()
        self.setupUi2ResetButton()
        self.setupUi2Label_1()
        self.setupUi2TextEdit()
        self.setupUi2DocumentButton()
        self.setupUi2Label_2()
        self.setupUi2ComboBox_1()
        self.setupUi2Label_3()
        self.setupUi2ComboBox_2()
        self.setupUi2ConfirmButton()
        self.setupUi2LoadLabel()

        """Ui界面3"""
        self.setupUi_3()
        self.setupPageWidget()
        self.setupUi3PushButtonLeft()
        self.setupUi3PushButtonRight()
        self.setupUi3PageEdit()
        self.setupUi3TotalPage()
        self.setupUi3ChenckBox()

        """Ui界面4"""
        self.setupUi_4()
        self.ui4TextEdit()
        self.ui4RecButton()
        self.setupUi4LoadLabel()
