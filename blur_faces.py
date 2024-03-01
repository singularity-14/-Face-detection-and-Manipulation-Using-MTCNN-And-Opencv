# Essential Libraries

# !pip install opencv-python
# !pip install mtcnn
import sklearn
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
from PIL import Image, ImageDraw,ImageFilter
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import os

# Dividing Images into 4 Different Segments
class ImageDivider(BaseEstimator, TransformerMixin):
    def transform(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        half_width, half_height = width // 2, height // 2

        top_left = img.crop((0, 0, half_width, half_height))
        top_right = img.crop((half_width, 0, width, half_height))
        bottom_left = img.crop((0, half_height, half_width, height))
        bottom_right = img.crop((half_width, half_height, width, height))

        return top_left, top_right, bottom_left, bottom_right

# Detecting the Faces and Defining the Bounding Box Around the Detected Face in Each Segment of Image
class FaceBlur(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.face_detector = MTCNN()

    def transform(self, quadrants):
        blurred_quadrants = []
        for i, quadrant in enumerate(quadrants):
            quadrant_cv = cv2.cvtColor(np.array(quadrant), cv2.COLOR_RGB2BGR)
            faces = self.face_detector.detect_faces(quadrant_cv)
            blurred_quadrants.append(self.blur_faces(quadrant, faces))
        return blurred_quadrants

    def blur_faces(self, image, faces):
        for face in faces: 
            x, y, width, height = face['box']
            aoi = image.crop((x, y, x + width, y + height))
            blur = aoi.filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR)
            image.paste(blur, (x, y, x + width, y + height))
        return image

# Combining the Each Segment of the Predcited Image and Formulating the Final Image
class ImageCombiner(BaseEstimator, TransformerMixin):
    def transform(self, blurred_quadrants):
        combined_image = Image.new('RGB', (blurred_quadrants[0].width + blurred_quadrants[1].width, blurred_quadrants[0].height + blurred_quadrants[2].height))
        combined_image.paste(blurred_quadrants[0], (0, 0))
        combined_image.paste(blurred_quadrants[1], (blurred_quadrants[0].width, 0))
        combined_image.paste(blurred_quadrants[2], (0, blurred_quadrants[0].height))
        combined_image.paste(blurred_quadrants[3], (blurred_quadrants[0].width, blurred_quadrants[0].height))
        return combined_image
    
# Function to Get a List of Image File-Paths in a Folder
def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png'] 
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(tuple(image_extensions))]
    return image_paths

# Input Folder and Output Folder
input_folder = r'/home/rachit/Documents/internship/services/canopy_cover/opencv/media'
output_folder = r'/home/rachit/Documents/internship/services/canopy_cover/opencv/outputs'

# Defining Pipelines
pipeline = Pipeline([
        ('divider', ImageDivider()),
        ('face_blur', FaceBlur()),
        ('combiner', ImageCombiner())
])


input_image_paths=get_image_paths(input_folder)

# Storing the Images in the Output Folder
for path in input_image_paths:
    result_image = pipeline.transform(path)

    output_image_path = os.path.join(output_folder, os.path.relpath(path, input_folder))

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    result_image.save(output_image_path)

print("Processing completed", output_folder)