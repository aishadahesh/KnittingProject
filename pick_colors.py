import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QColorDialog, QPushButton, QVBoxLayout, QWidget, QLabel,
    QHBoxLayout, QSpinBox, QGridLayout
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog

class ColorPickerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knitting Color Mapper")
        self.setGeometry(100, 100, 800, 600)

        self.selected_colors = []
        self.active_color_index = 0

        self.grid_buttons = []
        self.grid_data = []

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Color picker control
        control_layout = QHBoxLayout()
        self.color_spin = QSpinBox()
        self.color_spin.setMinimum(1)
        self.color_spin.setMaximum(10)
        self.color_spin.setValue(4)
        control_layout.addWidget(QLabel("Number of colors:"))
        control_layout.addWidget(self.color_spin)

        generate_btn = QPushButton("Generate Color Pickers")
        generate_btn.clicked.connect(self.generate_colors)
        control_layout.addWidget(generate_btn)
        main_layout.addLayout(control_layout)

        # Color buttons and grid layout
        self.content_layout = QHBoxLayout()
        self.color_buttons_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        self.content_layout.addLayout(self.color_buttons_layout)
        self.content_layout.addLayout(self.grid_layout)
        main_layout.addLayout(self.content_layout)

        # Save button
        save_btn = QPushButton("CLOSE")
        save_btn.clicked.connect(self.save_array)
        main_layout.addWidget(save_btn)

        self.setLayout(main_layout)

    def generate_colors(self):
        # Clear previous color buttons
        for i in reversed(range(self.color_buttons_layout.count())):
            widget = self.color_buttons_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.selected_colors = []
        self.grid_buttons = []
        self.grid_data = []

        num_colors = self.color_spin.value()
        for i in range(num_colors):
            color = QColorDialog.getColor()
            if color.isValid():
                rgba = (color.redF(), color.greenF(), color.blueF(), 1.0)
                self.selected_colors.append(rgba)

                btn = QPushButton()
                btn.setStyleSheet(f"background-color: {color.name()}; border: 2px solid black;")
                btn.setFixedSize(60, 60)
                btn.clicked.connect(lambda checked, idx=i: self.set_active_color(idx))
                self.color_buttons_layout.addWidget(btn)

        # Ask user for number of columns after colors are picked
        num_columns, ok = QInputDialog.getInt(
            self, "Number of Columns",
            "Enter number of columns for the bitmap grid:",
            min=1, max=20, value=4
        )
        if ok:
            self.build_grid(num_colors, num_columns)

    def set_active_color(self, index):
        self.active_color_index = index

    def build_grid(self, rows, cols):
        self.grid_data = [[1 for _ in range(cols)] for _ in range(rows)]  # Default to yellow (1)
        self.grid_buttons = [[None for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                btn = QPushButton()
                btn.setFixedSize(50, 50)
                btn.setStyleSheet("background-color: yellow;")
                btn.clicked.connect(lambda checked, x=i, y=j: self.toggle_cell(x, y))
                self.grid_layout.addWidget(btn, i, j)
                self.grid_buttons[i][j] = btn

    def toggle_cell(self, i, j):
        # Toggle between 1 (yellow) and 0 (black)
        self.grid_data[i][j] = 1 - self.grid_data[i][j]
        color = "yellow" if self.grid_data[i][j] == 1 else "black"
        self.grid_buttons[i][j].setStyleSheet(f"background-color: {color};")

    def save_array(self):
        array = np.array(self.grid_data)
        print("Grid (0=black, 1=yellow):\n", array)
        print("colors:\n", np.array(self.selected_colors))
        np.save("bitmap.npy", array)
        np.save("colors.npy", np.array(self.selected_colors))
        print("Saved as 'bitmap.npy' and 'colors.npy'")
        self.close() 


def run_color_app():
    app = QApplication(sys.argv)
    window = ColorPickerApp()
    window.show()
    app.exec()  # blocks until window closes
