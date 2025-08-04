import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import bpy


class RenderImagesApp(QWidget):
    def __init__(self, obj, render_callback):
        super().__init__()
        self.obj = obj
        self.render_callback = render_callback
        self.setWindowTitle("Render Knitting Images")
        self.setGeometry(100, 100, 500, 400)

        self.render_path = os.path.join(bpy.path.abspath("//images"), "result.png")
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.question_label = QLabel("Do you want to see rendered images?")
        self.question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.question_label)

        self.btn_yes = QPushButton("YES!!")
        self.btn_no = QPushButton("NO - Continue With Blender")

        self.btn_yes.clicked.connect(self.handle_yes)
        self.btn_no.clicked.connect(self.close)

        self.layout.addWidget(self.btn_yes)
        self.layout.addWidget(self.btn_no)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.image_label.setVisible(False)
        self.layout.addWidget(self.image_label)

        self.image_label2 = QLabel()
        self.image_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label2.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.image_label2.setVisible(False)
        self.layout.addWidget(self.image_label2)

        self.image_label3 = QLabel()
        self.image_label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label3.setStyleSheet("border: 2px dashed gray; padding: 10px;")
        self.image_label3.setVisible(False)
        self.layout.addWidget(self.image_label3)

        self.render_button = QPushButton("Render more Combinations")
        self.render_button.setVisible(False)
        self.render_button.clicked.connect(self.handle_rendering_more)
        self.layout.addWidget(self.render_button)

        self.close_button = QPushButton("Close")
        self.close_button.setVisible(False)
        self.close_button.clicked.connect(self.close)
        self.layout.addWidget(self.close_button)

        self.setLayout(self.layout)

    def handle_yes(self):
        self.question_label.hide()
        self.btn_yes.hide()
        self.btn_no.hide()

        if not os.path.exists(self.render_path):
            self.image_label.setText("Rendered image not found!")
        else:
            pixmap = QPixmap(self.render_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))

        self.image_label.setVisible(True)
        self.render_button.setVisible(True)
        self.close_button.setVisible(True)
        QMessageBox.information(self, "Rendering", "Rendered image displayed.")

    def handle_rendering_more(self):
        if self.render_callback and self.obj:
            self.render_callback(self.obj)
            render_path2 =  os.path.join(bpy.path.abspath("//images"), "combo_1.png")
            if not os.path.exists(render_path2):
                self.image_label2.setText("Rendered image not found!")
            else:
                pixmap = QPixmap(render_path2)
                self.image_label2.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.image_label2.setVisible(True)

            render_path3 =  os.path.join(bpy.path.abspath("//images"), "combo_2.png")
            if not os.path.exists(render_path3):
                self.image_label3.setText("Rendered image not found!")
            else:
                pixmap = QPixmap(render_path3)
                self.image_label3.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.image_label3.setVisible(True)
            QMessageBox.information(self, "Done", "More combinations rendered in /images/")
        else:
            QMessageBox.warning(self, "Error", "Render function or object missing.")
        self.render_button.setVisible(False)

def run_rendering_app(obj, render_callback):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = RenderImagesApp(obj, render_callback)
    window.show()
    app.exec()
