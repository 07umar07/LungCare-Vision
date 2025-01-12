# TB Lung Image Predictor

This is a Python-based graphical user interface (GUI) application for predicting the presence of Tuberculosis in lung images. The application leverages TensorFlow for model inference, PIL for image processing, and Tkinter for the GUI.

## Features

- **User-friendly Interface:** A simple and intuitive interface to load and analyze lung images.
- **TensorFlow Inference:** Utilizes TensorFlow Lite for fast and efficient model prediction.
- **Real-time Feedback:** Displays the lung image and prediction result immediately after selection.
- **Custom Styling:** Modern and visually appealing design using Tkinter's `ttk` widgets and themes.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/07umar07/Tuberculosis-Detector.git
    cd Tuberculosis-Detector
    ```

2. **Install the necessary dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` includes the following libraries:
    - numpy
    - tensorflow
    - pillow
    - tkinter (if not already included in your Python installation)

## Usage

1. **Run the application:**

    ```bash
    python main.py
    ```

2. **Using the interface:**
    - Click on the "Choose the Image" button to open a file dialog.
    - Select a lung image file from your directory.
    - The selected image and prediction result will be displayed on the GUI.

## Code Overview

- **main.py:** The main script that includes the GUI setup and prediction logic.

### Main Functions

- `output_predict(X)`: Takes in image data as input, processes it through the TensorFlow Lite model, and returns the prediction result.
- `load_image(img_path)`: Loads and preprocesses the image for model prediction.
- `on_drop()`: Handles the image selection process, updates the GUI with the image and prediction result.

### GUI Setup

The GUI is created using Tkinter's `ttk` widgets and styled with a custom theme for a modern look. It includes a button to choose an image, and frames to display the selected image and prediction result.

## Screenshots

![Screenshot 1](path_to_screenshot_1)

![Screenshot 2](path_to_screenshot_2)

## Contributing

Feel free to submit issues, fork the repository, and make pull requests. Any contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
