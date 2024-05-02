# Face Detection and Manipulation Using MTCNN And OpenCV

## Overview

This project implements a MultiTask Convolutional Neural Network (MTCNN) along with OpenCV to detect and blur faces in images obtained from a city survey. It divides the images into four segments, detects faces in each segment, and applies a blur filter to the detected faces. Finally, it combines the segments to form the final image with blurred faces.

## Essential Libraries

- **OpenCV**: Required for image processing tasks.
- **MTCNN**: Multi-Task Convolutional Neural Network for face detection.
- **scikit-learn**: Used for constructing pipelines.
- **Pillow (PIL)**: Python Imaging Library for image manipulation.
- **NumPy**: For numerical operations on arrays.

## Usage

1. **Installation**: Install the required libraries using pip:
   ```
   pip install opencv-python mtcnn scikit-learn pillow numpy
   ```

2. **Input and Output Folders**:
   - Define the input folder containing the images to be processed.
   - Define the output folder where the processed images will be saved.

3. **Running the Script**:
   - Run the script `face_detection_and_blur.py`.
   - Ensure that the input folder path is correctly specified within the script.
   - The processed images will be saved in the output folder.

## Customization

- You can modify the blur intensity or apply different filters in the `FaceBlur` class.
- Adjust the size or number of segments in the `ImageDivider` class based on your requirements.

## File Structure

- `face_detection_and_blur.py`: Main Python script for face detection and blur.
- `README.md`: This file providing an overview and instructions.
- `input_folder/`: Folder containing input images.
- `output_folder/`: Folder where processed images will be saved.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Rachit Patel 
