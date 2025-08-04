import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QColorDialog, QPushButton, QVBoxLayout, QWidget, QLabel,
    QHBoxLayout, QSpinBox, QGridLayout, QMessageBox
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt


class ColorPickerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knitting Color Mapper")
        self.setGeometry(100, 100, 900, 600)

        self.selected_colors = [
            (1.0, 1.0, 0.0, 1.0),  # yellow
            (0.0, 0.0, 0.0, 1.0),  # black
            (1.0, 0.0, 0.0, 1.0),  # red
            (0.0, 0.0, 1.0, 1.0),  # blue
        ]
        self.active_color_index = 0

        self.cols = 4
        self.grid_buttons = []
        self.grid_data = []

        # colors for grid toggle
        self.grid_toggle_colors = [
            (0.0, 0.0, 0.0, 1.0),  # black
            (1.0, 1.0, 0.0, 1.0),  # yellow
        ]

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Color controls
        color_control_layout = QHBoxLayout()
        color_control_layout.addWidget(QLabel("Right-click color to edit, Left-click to delete:"))

        self.color_buttons_layout = QHBoxLayout()
        color_control_layout.addLayout(self.color_buttons_layout)
        main_layout.addLayout(color_control_layout)

        self.refresh_color_buttons()

        # Grid size controls
        size_control_layout = QHBoxLayout()
        size_control_layout.addWidget(QLabel("Columns:"))
        self.col_spin = QSpinBox()
        self.col_spin.setMinimum(1)
        self.col_spin.setValue(self.cols)
        self.col_spin.valueChanged.connect(self.update_grid_size)
        size_control_layout.addWidget(self.col_spin)
        main_layout.addLayout(size_control_layout)

        # Grid display
        self.grid_layout = QGridLayout()
        main_layout.addLayout(self.grid_layout)

        self.rebuild_grid()

        # Save button
        save_btn = QPushButton("See Result")
        save_btn.clicked.connect(self.save_array)
        main_layout.addWidget(save_btn)

        self.setLayout(main_layout)

    def refresh_color_buttons(self):
        # Clear existing buttons
        for i in reversed(range(self.color_buttons_layout.count())):
            widget = self.color_buttons_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Add color buttons with click handlers
        for i, rgba in enumerate(self.selected_colors):
            btn = QPushButton()
            btn.setFixedSize(40, 40)
            color = QColor.fromRgbF(*rgba)
            btn.setStyleSheet(f"background-color: {color.name()}; border: 2px solid black;")
            btn.setToolTip("Right-click to edit. Left-click to delete.")
            btn.mousePressEvent = self.make_mouse_event_handler(i)
            self.color_buttons_layout.addWidget(btn)

        # Add "+" button at the end
        add_color_button = QPushButton("+")
        add_color_button.setFixedSize(30, 30)
        add_color_button.setStyleSheet("background-color: lightgray; font-weight: bold; border: 2px solid gray;")
        add_color_button.setToolTip("Add new color")
        add_color_button.clicked.connect(self.add_color)
        self.color_buttons_layout.addWidget(add_color_button)

    def make_mouse_event_handler(self, index):
        def handler(event):
            if event.button() == Qt.MouseButton.LeftButton:
                # Left click = delete color
                if len(self.selected_colors) <= 1:
                    QMessageBox.warning(self, "Warning", "You must have at least one color.")
                    return
                del self.selected_colors[index]
                self.refresh_color_buttons()
                self.rebuild_grid()
            elif event.button() == Qt.MouseButton.RightButton:
                # Right click = edit color
                self.edit_color(index)
        return handler

    def add_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            rgba = (color.redF(), color.greenF(), color.blueF(), 1.0)
            self.selected_colors.append(rgba)
            self.refresh_color_buttons()
            self.rebuild_grid()

    def edit_color(self, index):
        self.active_color_index = index
        color = QColorDialog.getColor()
        if color.isValid():
            rgba = (color.redF(), color.greenF(), color.blueF(), 1.0)
            self.selected_colors[index] = rgba
            self.refresh_color_buttons()
            self.rebuild_grid()

    def update_grid_size(self):
        self.cols = self.col_spin.value()
        self.rebuild_grid()

    def rebuild_grid(self):
        rows = len(self.selected_colors)
        cols = self.cols

        # Clear previous grid widgets
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Initialize grid data to all yellow (index 1)
        self.grid_data = [[1 for _ in range(cols)] for _ in range(rows)]
        self.grid_buttons = []

        for i in range(rows):
            row_buttons = []
            for j in range(cols):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                rgba = self.grid_toggle_colors[self.grid_data[i][j]]  
                color = QColor.fromRgbF(*rgba)
                btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
                btn.clicked.connect(lambda checked, x=i, y=j: self.toggle_cell_color(x, y))
                self.grid_layout.addWidget(btn, i, j)
                row_buttons.append(btn)
            self.grid_buttons.append(row_buttons)

    def toggle_cell_color(self, i, j):
        current_val = self.grid_data[i][j]
        # toggle between 1 (yellow) and 0 (black)
        new_val = 0 if current_val == 1 else 1
        self.grid_data[i][j] = new_val
        rgba = self.grid_toggle_colors[new_val]  
        color = QColor.fromRgbF(*rgba)
        self.grid_buttons[i][j].setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

    def save_array(self):
        array = np.array(self.grid_data)
        print("Grid:\n", array)
        print("Colors:\n", np.array(self.selected_colors))
        np.save("bitmap.npy", array)
        np.save("colors.npy", np.array(self.selected_colors))
        print("Saved to 'bitmap.npy' and 'colors.npy'")
        self.close()


def run_color_app():
    app = QApplication(sys.argv)
    window = ColorPickerApp()
    window.show()
    app.exec()


# if __name__ == "__main__":
#     run_color_app()
