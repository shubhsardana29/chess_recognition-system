from typing import Optional
from PyQt5.QtWidgets import QMainWindow, QLabel, QToolTip
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# GUI class for displaying the main interface
class GUI(QMainWindow):
    def __init__(self, title: str = 'preview'):
        super(GUI, self).__init__(parent=None)

        self.__window_size = (440, 440)
        self.__console_texts: list[str] = []
        self.__max_buffer_size = 10

        # Initialize the canvas (Graphics Layout Widget)
        self.__canvas = pg.GraphicsLayoutWidget(size=self.__window_size)
        self.__canvas.setWindowTitle(title)

        # Initialize layout
        self.__layout = pg.GraphicsLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__canvas.setCentralItem(self.__layout)

        # Add view and image
        self.__view = pg.ViewBox(enableMouse=False)
        self.__view.suggestPadding = lambda *_: 0.0
        self.__view.invertY()
        self.__layout.addItem(self.__view)

        self.__image_item = pg.ImageItem(axisOrder='row-major')
        self.__view.addItem(self.__image_item)

        # Setup console log
        self.__createConsole()

        # Define tooltip settings
        QToolTip.setFont(QFont('Helvetica', 18))

    def __createConsole(self):
        self.label = QLabel(self.__canvas)
        self.label.setStyleSheet('QLabel { color: yellow; margin: 10px; font-weight: bold }')

    def __showConsoleText(self):
        self.label.setText('\n'.join(self.__console_texts))
        self.label.adjustSize()

    def setImage(self, img):
        self.__image_item.setImage(img)

    def print(self, text: str = '', index: Optional[int] = None):
        if index is None:
            self.__console_texts.append(text)
        else:
            if len(self.__console_texts) > 0:
                self.__console_texts.pop(index)
            self.__console_texts.insert(index, text)

        if len(self.__console_texts) > self.__max_buffer_size:
            self.__console_texts.pop(0)

        self.__showConsoleText()

    def show(self):
        """Show application window"""
        self.__canvas.show()

