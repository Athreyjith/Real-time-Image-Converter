import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Ensure output folder exists
output_folder = r'C:\Users\athre\Desktop\1GL PROGRAM11\output'
os.makedirs(output_folder, exist_ok=True)

# Subdirectories for each effect
effects = ['Blur', 'Gray', 'Sketch', 'Paint']
for effect in effects:
    os.makedirs(os.path.join(output_folder, effect), exist_ok=True)

# Function to create Blur animation effect by shifting alternate rows
def blur_animation(image):
    combed_image = image.copy()
    rows, cols = combed_image.shape[:2]
    for i in range(0, rows, 2):
        combed_image[i] = np.roll(combed_image[i], i % 20)
    return combed_image

# Function to convert the image to a sketch effect
def sketch_effect(image):
    inverted_image = cv2.bitwise_not(image)
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(image, inverted_blurred, scale=256.0)
    return sketch

# Function to create a paint effect
def paint_effect(image):
    data = np.float32(image).reshape((-1, 3))
    _, label, center = cv2.kmeans(data, 8, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    reduced_color_image = result.reshape(image.shape)
    painted_image = cv2.bilateralFilter(reduced_color_image, d=9, sigmaColor=75, sigmaSpace=75)
    return painted_image

# Function to generate a unique filename
def get_unique_filename(base_path, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_path}_{timestamp}.{extension}"

# Function to process and save images from file paths
def process_and_save_images(image_paths):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to load: {image_path}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        combed_image = blur_animation(image)
        sketched_image = sketch_effect(gray_image)
        painted_image = paint_effect(image)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save images to subdirectories
        cv2.imwrite(os.path.join(output_folder, 'Blur', get_unique_filename(base_filename, 'jpg')), combed_image)
        cv2.imwrite(os.path.join(output_folder, 'Gray', get_unique_filename(base_filename, 'jpg')), gray_image)
        cv2.imwrite(os.path.join(output_folder, 'Sketch', get_unique_filename(base_filename, 'jpg')), sketched_image)
        cv2.imwrite(os.path.join(output_folder, 'Paint', get_unique_filename(base_filename, 'jpg')), painted_image)

        # Save original image to main output folder
        cv2.imwrite(os.path.join(output_folder, get_unique_filename(base_filename, 'jpg')), image)

        plt.figure(figsize=(25, 10))
        plt.subplot(1, 6, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 6, 2)
        plt.imshow(cv2.cvtColor(combed_image, cv2.COLOR_BGR2RGB))
        plt.title('Blur Animation')
        plt.axis('off')

        plt.subplot(1, 6, 3)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Greyscale effect')
        plt.axis('off')

        plt.subplot(1, 6, 4)
        plt.imshow(sketched_image, cmap='gray')
        plt.title('Sketch Effect')
        plt.axis('off')

        plt.subplot(1, 6, 5)
        plt.imshow(cv2.cvtColor(painted_image, cv2.COLOR_BGR2RGB))
        plt.title('Paint Effect')
        plt.axis('off')

        plt.show()

# Function to capture and process image from webcam
def webcam_capture_and_process():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key is pressed
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            combed_image = blur_animation(frame)
            sketched_image = sketch_effect(gray_image)
            painted_image = paint_effect(frame)

            base_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save images to subdirectories
            cv2.imwrite(os.path.join(output_folder, 'Blur', get_unique_filename(base_filename, 'jpg')), combed_image)
            cv2.imwrite(os.path.join(output_folder, 'Gray', get_unique_filename(base_filename, 'jpg')), gray_image)
            cv2.imwrite(os.path.join(output_folder, 'Sketch', get_unique_filename(base_filename, 'jpg')), sketched_image)
            cv2.imwrite(os.path.join(output_folder, 'Paint', get_unique_filename(base_filename, 'jpg')), painted_image)

            # Save original image to main output folder
            cv2.imwrite(os.path.join(output_folder, get_unique_filename(base_filename, 'jpg')), frame)

            plt.figure(figsize=(25, 10))
            plt.subplot(1, 6, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 6, 2)
            plt.imshow(cv2.cvtColor(combed_image, cv2.COLOR_BGR2RGB))
            plt.title('Blur Animation')
            plt.axis('off')

            plt.subplot(1, 6, 3)
            plt.imshow(gray_image, cmap='gray')
            plt.title('Greyscale effect')
            plt.axis('off')

            plt.subplot(1, 6, 4)
            plt.imshow(sketched_image, cmap='gray')
            plt.title('Sketch Effect')
            plt.axis('off')

            plt.subplot(1, 6, 5)
            plt.imshow(cv2.cvtColor(painted_image, cv2.COLOR_BGR2RGB))
            plt.title('Paint Effect')
            plt.axis('off')

            plt.show()
            break

        elif key == ord('q'):  # Press 'q' to quit without saving
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script
def main():
    print("Choose an option:")
    print("1: Process images from files")
    print("2: Capture and process image from webcam")

    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        image_paths = [
            r'C:\Users\athre\Desktop\1GL PROGRAM11\input\download.jpg',
            r'C:\Users\athre\Desktop\1GL PROGRAM11\input\pexels-photo.jpg',
            r'C:\Users\athre\Desktop\1GL PROGRAM11\input\panthera-tigris-altaica-tiger-siberian-amurtiger-162203.jpeg'
        ]
        process_and_save_images(image_paths)

    elif choice == '2':
        webcam_capture_and_process()

    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
