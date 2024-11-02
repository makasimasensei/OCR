import sys

from PyQt5.QtWidgets import QApplication
from ui.ui_widgets import *
from PyQt5.QtCore import Qt


class Window(Ui_Widget):
    def __init__(self):
        super().__init__()
        self.setupUi()


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = Window()
    sys.exit(app.exec_())
