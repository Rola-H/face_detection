from mtcnn import MTCNN
import cv2

def face_detect(image_path)
    # Load the MTCNN face detector
    detector = MTCNN()
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    
    # Define the desired width
    desired_width = 900
    
    # Calculate the aspect ratio
    aspect_ratio = desired_width / image.shape[0]
    
    # Calculate the corresponding height while maintaining the aspect ratio
    desired_height = int(image.shape[1] * aspect_ratio)
    
    # Resize the image
    image = cv2.resize(image, (desired_height, desired_width))
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform face detection
    faces = detector.detect_faces(image_rgb)
    
    # Draw rectangles on the detected faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image
