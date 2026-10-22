import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

from Chinook_Customer_Clustering.Ui.ChinookMainWindowEx import ChinookMainWindowEx

app = QApplication(sys.argv)
main_window = QMainWindow()
myui = ChinookMainWindowEx()
myui.setupUi(main_window)
myui.showWindow()

sys.exit(app.exec())
