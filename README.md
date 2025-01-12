# TB Lung Image Predictor

This is a Python-based graphical user interface (GUI) application for predicting the presence of Tuberculosis in X-ray lung images. The application leverages TensorFlow for model inference, PIL for image processing, and Tkinter for the GUI.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Demonstration Video](#Demonstration-Video)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [References](#references)

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
    - matplotlib
    - imblearn
    - scikit-learn
    - pillow
    - tkinter (if not already included in your Python installation)

## Usage

1. **Open Jupyter Notebook or Google Colab:**

    ```bash
    jupyter notebook
    ```

2. **Run the notebook:**
    - Navigate to the directory where the `.ipynb` file is located.
    - Open the notebook in your browser.
    - Run the cells in the notebook to execute the application.

3. **Using the interface:**
    - Click on the "Choose the Image" button to open a file dialog.
    - Select a lung image file from your directory.
    - The selected image and prediction result will be displayed on the GUI.

## Code Overview

- **TBCDetector.ipynb:** The main script that includes model training and testing.
- **Test.ipynb:** The testing script includes the GUI setup and prediction test.

### Main Functions

- `output_predict(X)`: Takes in image data as input, processes it through the TensorFlow Lite model, and returns the prediction result.
- `load_image(img_path)`: Loads and preprocesses the image for model prediction.
- `on_drop()`: Handles the image selection process, updates the GUI with the image and prediction result.

### GUI Setup

The GUI is created using Tkinter's `ttk` widgets and styled with a custom theme for a modern look. It includes a button to choose an image, and frames to display the selected image and prediction result.

## Demonstration Video

Watch the demonstration video below to see the TB Lung Image Predictor in action:
[![Watch the video](https://img.youtube.com/vi/OfoCx7cwaSk/0.jpg)](https://www.youtube.com/watch?v=OfoCx7cwaSk)


## Contributing

Feel free to submit issues, fork the repository, and make pull requests. Any contributions are welcome!

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Citation

Researchers can use this database to produce useful and impactful scholarly work on TB, which can help in tackling this issue.

Please cite this database if you are using it for any scientific purpose:
Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020) "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization". IEEE Access, Vol. 8, pp 191586 - 191601. DOI: 10.1109/ACCESS.2020.3031384.

## References

1. S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases," Quantitative imaging in medicine and surgery, vol. 4 (6), p. 475 (2014)
2. B. P. Health. (2020). BELARUS TUBERCULOSIS PORTAL [Online]. Available: [http://tuberculosis.by/](http://tuberculosis.by/). [Accessed on 09-June-2020]
3. NIAID TB portal program dataset [Online]. Available: [https://tbportals.niaid.nih.gov/download-data](https://tbportals.niaid.nih.gov/download-data).
4. kaggle. Tuberculosis (TB) Chest X-ray Database [Online]. Available: [https://www.kaggle.com/datasets/scipygaurav/tuberculosis-tb-chest-x-ray-cleaned-database](https://www.kaggle.com/datasets/scipygaurav/tuberculosis-tb-chest-x-ray-cleaned-database).
