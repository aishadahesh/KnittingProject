import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
import bpy
import glob

class RenderImagesApp(QWidget):
    def __init__(self, obj, render_callback, other_window=None):
        super().__init__()
        self.obj = obj
        self.render_callback = render_callback
        self.other_window = other_window
        self.setWindowTitle("Render Knitting Images")
        self.setGeometry(100, 100, 500, 600)

        self.render_path = os.path.join(bpy.path.abspath("//images"), "result.png")
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 10px;")
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

        self.see_more_button = QPushButton("See More")
        self.see_more_button.clicked.connect(self.show_more_images)
        self.layout.addWidget(self.see_more_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_both_windows)
        self.layout.addWidget(self.close_button)

        self.setLayout(self.layout)

    def clear_images_folder(self, folder_path):
        pattern = os.path.join(folder_path, "combo_*.png")
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

    def refresh_render(self):
        images_folder = bpy.path.abspath("//images")
        self.clear_images_folder(images_folder)  

        if self.obj and self.render_callback:
            self.render_callback(self.obj)

        def load_first_image():
            if os.path.exists(self.render_path):
                pixmap = QPixmap(self.render_path)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
            else:
                self.image_label.setText("Rendered image not found!")

            self.image_label2.setVisible(False)
            self.image_label3.setVisible(False)
            self.see_more_button.setVisible(True)

        # Delay loading first image by 200ms to ensure file exists
        QTimer.singleShot(200, load_first_image)

    def show_more_images(self):
        combo1 = os.path.join(bpy.path.abspath("//images"), "combo_1.png")
        combo2 = os.path.join(bpy.path.abspath("//images"), "combo_2.png")

        if os.path.exists(combo1):
            pixmap = QPixmap(combo1)
            self.image_label2.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.image_label2.setVisible(True)
        else:
            self.image_label2.setText("Combo_1 not found")
            self.image_label2.setVisible(False)

        if os.path.exists(combo2):
            pixmap = QPixmap(combo2)
            self.image_label3.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
            self.image_label3.setVisible(True)
        else:
            self.image_label3.setText("Combo_2 not found")
            self.image_label3.setVisible(False)

        self.see_more_button.setVisible(False)

    def close_both_windows(self):
        self.close()
        if self.other_window is not None:
            self.other_window.close()


def run_rendering_app(obj, render_callback, other_window=None):
    return RenderImagesApp(obj, render_callback, other_window)
