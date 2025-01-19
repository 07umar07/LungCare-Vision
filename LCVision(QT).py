import sys
import numpy as np
import onnxruntime as ort
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSpacerItem,
    QFileDialog, QWidget, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

# Created by Umar Robbani
# This is the code for implement Machine Learning model with QT frameworks for the GUI.

# Load ONNX model
MODEL_PATH = '/home/umar/myenv/SelfProject/TBCMLModels/Model/ONNX/FINAL_MODEL.onnx'
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict_tuberculosis(image_array):
    prediction = session.run([output_name], {input_name: image_array.astype(np.float32)})[0].ravel()[0]
    if prediction < 0.5:
        return f"The lung looks good\nPercentage of Tuberculosis: {prediction * 100:.2f}%", '#33a933'
    return f"High chance of Tuberculosis\nPercentage of Tuberculosis: {prediction * 100:.2f}%",'#c73333' 

def preprocess_image(file_path):
    img = Image.open(file_path).convert('L').resize((224, 224))
    return np.array(img).reshape(1, 224, 224, 1) / 255.0

class DraggableLabel(QLabel):
    # Drag and Drop Functionality
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_url = event.mimeData().urls()[0].toLocalFile()
            if file_url.lower().endswith('.png'):
                self.parent().process_image(file_url)
            else:
                self.setText("Please drop a valid PNG image.")
        else:
            event.ignore()

class TBDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LungCare Vision")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
                font-family: Arial, sans-serif;
                font-size: 18px
            }
            QLabel {
                color: #333;
                font-size: 24px;
            }
            QPushButton {
                background-color: #5A9;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 24px;
            }
            QPushButton:hover {
                background-color: #48a;
            }
            QLabel#footer {
                color: #666;
                font-size: 15px;
            }
        """)

        # Layouts
        self.layout = QVBoxLayout()
        self.result_frame = QFrame(self)
        self.result_layout = QVBoxLayout(self.result_frame)
        self.result_frame.hide()  # Ensure the result frame is hidden initially

        # Title
        self.title = QLabel("Please provide the lung X-ray image for analysis.", self)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.title)
        self.layout.addItem(QSpacerItem(10, 20))

        # Choose Button
        self.choose_button = QPushButton("Choose the Image", self)
        self.choose_button.clicked.connect(self.choose_image)
        self.layout.addWidget(self.choose_button)
        self.layout.addItem(QSpacerItem(10, 10))

        # Drag and Drop Area with Result
        self.drag_label = DraggableLabel(self)
        self.drag_label.setText("Or drag and drop PNG image here")

        self.drag_label.setAlignment(Qt.AlignCenter)
        self.drag_label.setStyleSheet("border: 2px dashed #ccc; padding: 60px; color: #666; font-size: 24px")
        self.layout.addWidget(self.drag_label)

        # Image Display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("margin-top: 20px;")
        self.layout.addWidget(self.image_label)

        # Result Label
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.layout.addWidget(self.result_label)
        spacer = QSpacerItem(20, 20)
        self.layout.addItem(spacer)

        # Back Button
        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.reset_interface)
        self.layout.addWidget(self.back_button)
        self.back_button.hide() # Ensure back button is hide initially

        # Spacer to bully the footer so it would always be in the bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer) 

        # Footer
        footer_layout = QHBoxLayout()
        footer_text = QLabel("© 2025 Umar Robbani - Licensed under GPLv3", self)
        footer_text.setObjectName("footer")
        footer_layout.addWidget(footer_text)
        footer_layout.addStretch(1)
        github_label = QLabel('<a href="https://github.com/07umar07/Tuberculosis-Detector">GitHub</a>', self)
        github_label.setOpenExternalLinks(True)
        github_label.setStyleSheet("color: #1E90FF;")
        footer_layout.addWidget(github_label)
        version_label = QLabel("v1.0.0", self)
        version_label.setObjectName("footer")
        footer_layout.addWidget(version_label)
        self.layout.addLayout(footer_layout)

        self.setLayout(self.layout)

    def choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "PNG Images (*.png)")
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        try:
            img_array = preprocess_image(file_path)
            result_text, color = predict_tuberculosis(img_array)
            self.display_result(file_path, result_text, color)
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")

    def display_result(self, file_path, result_text, color):
        # Show image
        img = Image.open(file_path).resize((224, 224)).convert('L')
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

        # Show result
        self.result_label.setText(result_text)
        self.result_label.setStyleSheet(f"color: {color};") # Change the font color of result dynamically
        self.result_frame.show()  # Show the result frame when result is available
        self.back_button.show() # Show the back button when result is available
        self.title.hide()
        self.choose_button.hide()
        self.drag_label.hide()

    def reset_interface(self):
        self.result_label.clear()
        self.result_label.setStyleSheet("")
        self.image_label.clear()
        self.back_button.hide()
        self.title.show()
        self.drag_label.show()
        self.choose_button.show()

# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TBDetectorApp()
    window.show()
    sys.exit(app.exec_())
