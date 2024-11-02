from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QTextEdit


class MyTextEdit(QTextEdit):
    returnPressed = pyqtSignal()  # 自定义信号
    fileDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 接受拖放
        self.setAcceptDrops(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.returnPressed.emit()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_paths = []
        for url in event.mimeData().urls():
            file_paths.append(url.toLocalFile())
        self.fileDropped.emit(file_paths)
