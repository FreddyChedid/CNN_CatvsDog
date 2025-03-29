import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QSpacerItem, QSizePolicy, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time  # To simulate delay for loading

class CatDogClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the pre-trained model
        self.model = load_model('model_cat_vs_dog.h5')  # Path to your model
        
        # Setup the UI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Cat vs Dog Classifier')
        self.setFixedSize(600, 600)  # Set a fixed window size
        self.setStyleSheet('background-color: #f0f0f0; font-family: Arial, sans-serif;')
        
        # Create main layout
        self.main_layout = QVBoxLayout()

        # Header: Prediction Title
        self.header_label = QLabel('Cat vs Dog Image Classifier', self)
        self.header_label.setStyleSheet('font-size: 28px; font-weight: bold; color: #2C3E50;')
        self.header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header_label)

        # Spacer
        self.main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Image display label
        self.image_label = QLabel('No image selected', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet('border: 3px solid #BDC3C7; padding: 10px; background-color: #ECF0F1; border-radius: 10px;')
        self.main_layout.addWidget(self.image_label)

        # Spacer
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Load Image button
        self.load_button = QPushButton('Load Image', self)
        self.load_button.setStyleSheet('background-color: #3498DB; color: white; font-size: 16px; padding: 12px; border-radius: 10px;')
        self.load_button.clicked.connect(self.load_image)
        self.main_layout.addWidget(self.load_button)

        # Spacer
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498DB;
                border-radius: 5px;
                text-align: center;
                background-color: #ECF0F1;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                width: 10px;
            }
        """)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.main_layout.addWidget(self.progress_bar)

        # Spacer
        self.main_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Result label
        self.result_label = QLabel('Prediction: None', self)
        self.result_label.setStyleSheet('font-size: 20px; font-weight: bold; color: #7F8C8D;')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.result_label)

        # Set main layout for the window
        self.setLayout(self.main_layout)
        self.show()

    def load_image(self):
        # Open file dialog to select an image
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp *.jpeg)')
        
        if file_name:
            # Load the image and display it
            self.display_image(file_name)
            # Start the prediction process
            self.predict_image(file_name)

    def display_image(self, file_name):
        # Load image using QPixmap for display
        img = QPixmap(file_name)
        img = img.scaled(400, 400, Qt.KeepAspectRatio)  # Scale while keeping aspect ratio
        self.image_label.setPixmap(img)
        self.image_label.setText('')  # Clear the text

    def predict_image(self, file_name):
        # Show progress bar and simulate a delay for loading
        self.progress_bar.setValue(10)
        self.result_label.setText('Prediction: Processing...')
        
        # Simulate a loading process
        time.sleep(1)  # Simulate some processing time (remove it in production)

        # Preprocess the image for prediction
        img = image.load_img(file_name, target_size=(128, 128))  # Resize to model's input size
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Simulate progress while predicting
        self.progress_bar.setValue(50)
        
        # Make prediction using the model
        prediction = self.model.predict(img_array)
        
        # Simulate progress after prediction
        self.progress_bar.setValue(90)
        
        # Display the result in the GUI
        if prediction[0][0] > 0.5:
            self.result_label.setText('Prediction: Dog')
            self.result_label.setStyleSheet('font-size: 20px; font-weight: bold; color: #E74C3C;')
        else:
            self.result_label.setText('Prediction: Cat')
            self.result_label.setStyleSheet('font-size: 20px; font-weight: bold; color: #2ECC71;')

        # Finalize progress bar to 100%
        self.progress_bar.setValue(100)
        self.progress_bar.setTextVisible(False)  # Hide text when done

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CatDogClassifierApp()
    sys.exit(app.exec_())
