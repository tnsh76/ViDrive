from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import logging
from flask import Flask, request, render_template, redirect, url_for, flash,send_from_directory
import os
import torch
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
import time
import matplotlib.pyplot as plt

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['CHART_FOLDER'] = 'static/charts'
app.secret_key = 'your_secret_key'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to the models
model_dir = 'models'
model1_path = os.path.join(model_dir, 'full_driver_distraction_model_resnet_finetune.pth')
model2_path = os.path.join(model_dir, 'driver_distraction_model2.h5')

# Load Model 1 (PyTorch model) with map_location to handle CPU-only devices
model1 = torch.load(model1_path, map_location=device)
model1.to(device)

# Load Model 2 (TensorFlow model)
model2 = tf.keras.models.load_model(model2_path)

# Define distraction classes for each model
model1_classes = ['normal driving', 'texting - right', 'talking on the phone - right', 
                  'texting - left', 'talking on the phone - left', 'operating the radio', 
                  'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']

model2_classes = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

# Define test transform for Model 1
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict distraction using Model 1
def predict_distraction_model1(image_path, model):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    predicted_class = model1_classes[probabilities.argmax().item()]
    confidence = probabilities.max().item()
    
    return predicted_class, confidence

# Function to preprocess image for Model 2
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict using Model 2
def predict_model2(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model2.predict(processed_image)
    class_id = np.argmax(prediction)
    
    if class_id < len(model2_classes):
        class_name = model2_classes[class_id]
    else:
        class_name = f"Unknown (Class ID: {class_id})"
    
    confidence = float(prediction[0][class_id])
    return class_name, confidence

# Combined prediction function
def combined_prediction(image_path):
    try:
        # Prediction from Model 1
        class_name1, confidence1 = predict_distraction_model1(image_path, model1)
    except Exception as e:
        print(f"Error in Model 1 prediction: {str(e)}")
        class_name1, confidence1 = "Error", 0.0

    try:
        # Prediction from Model 2
        class_name2, confidence2 = predict_model2(image_path)
    except Exception as e:
        print(f"Error in Model 2 prediction: {str(e)}")
        class_name2, confidence2 = "Error", 0.0

    # Determine the model with the higher confidence
    if confidence1 >= confidence2:
        print(f"Model with higher confidence: Model1 ({confidence1}) - Class: {class_name1}")
    else:
        print(f"Model with higher confidence: Model2 ({confidence2}) - Class: {class_name2}")

    return {
        'Model1': {'class': class_name1, 'confidence': confidence1},
        'Model2': {'class': class_name2, 'confidence': confidence2}
    }

# Load your trained model
model_path = 'models/full_driver_distraction_model_resnet_finetune.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the class names
class_names = ['normal driving', 'texting - right', 'talking on the phone - right', 'texting - left',
               'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
               'hair and makeup', 'talking to passenger']

def detect_driver_distraction(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            label = class_names[preds[0]]
            results.append(label)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        time.sleep(0.1)  # Adjust the delay as needed

        # Display the frame
        cv2.imshow('Driver Distraction Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results

def create_bar_chart(results):
    # Count the frequency of each class in the results
    class_counts = {class_name: results.count(class_name) for class_name in class_names}

    # Generate the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Driver Distraction Detection Results')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the bar chart as an image
    chart_path = os.path.join(app.config['CHART_FOLDER'], 'detection_results.png')
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# Global variables

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Thresholds and Counters
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 45
HEAD_POSE_THRESH = 25
DISTRACTED_ALERT_THRESH = 6
DROWSY_ALERT_THRESH = 3
YAWNING_ALERT_THRESH = 1
MAR_HISTORY_LENGTH = 10
MAR_THRESH_MULTIPLIER = 0.95
YAWN_COUNT_THRESH = 5

COUNTER = 0
DROWSY_COUNT = 0
YAWN_COUNTER = 0
DISTRACTED_COUNT = 0

mar_history = []
yawning_detected = False
drowsiness_alert_counter = 0

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Global variable to control video capture
capture = False

def speak(message):
    print(f"Speaking: {message}")
    engine.say(message)
    engine.runAndWait()
    print("Speech should have been output.")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(np.array(mouth[3]) - np.array(mouth[9]))
    B = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[10]))
    C = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[8]))
    D = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))
    mar = (A + B + C) / (3.0 * D)
    return mar

def calculate_head_pose(image, landmarks):
    size = image.shape
    image_points = np.array([
        (landmarks[1].x * size[1], landmarks[1].y * size[0]),  # Nose tip
        (landmarks[152].x * size[1], landmarks[152].y * size[0]),  # Chin
        (landmarks[263].x * size[1], landmarks[263].y * size[0]),  # Left eye left corner
        (landmarks[33].x * size[1], landmarks[33].y * size[0]),  # Right eye right corner
        (landmarks[287].x * size[1], landmarks[287].y * size[0]),  # Left Mouth corner
        (landmarks[57].x * size[1], landmarks[57].y * size[0])  # Right Mouth corner
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right Mouth corner
    ])
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return p1, p2, rotation_vector, translation_vector


@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture
    capture = True
    return ('', 204)

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capture
    capture = False
    return ('', 204)

def gen_frames():
    cap = cv2.VideoCapture(0)
    global COUNTER, DROWSY_COUNT, YAWN_COUNTER, DISTRACTED_COUNT, mar_history, yawning_detected, drowsiness_alert_counter

    while cap.isOpened():
        if not capture:
            continue

        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                h, w, _ = image.shape

                # Get the coordinates of the eye landmarks
                left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
                left_eye = [(int(landmark.x * w), int(landmark.y * h)) for landmark in left_eye]
                right_eye = [(int(landmark.x * w), int(landmark.y * h)) for landmark in right_eye]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Get the coordinates of the mouth landmarks
                mouth = [landmarks[i] for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 306]]
                mouth = [(int(landmark.x * w), int(landmark.y * h)) for landmark in mouth]
                mar = mouth_aspect_ratio(mouth)

                # Calculate head pose
                p1, p2, rotation_vector, translation_vector = calculate_head_pose(image, landmarks)
                head_angle_x = np.degrees(rotation_vector[0])
                head_angle_y = np.degrees(rotation_vector[1])
                head_angle_z = np.degrees(rotation_vector[2])

                # Eye aspect ratio (EAR) detection
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(image, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        DROWSY_COUNT += 1
                        if DROWSY_COUNT >= DROWSY_ALERT_THRESH and yawning_detected:
                            drowsiness_alert_counter += 1
                            DROWSY_COUNT = 0
                            yawning_detected = False
                            print("Drowsiness detected (counter incremented).")
                            if drowsiness_alert_counter >= 3:
                                speak("Drowsiness detected, please stay alert on the road.")
                                drowsiness_alert_counter = 0
                                print("Drowsiness detected, alert spoken.")
                else:
                    COUNTER = 0

                # Dynamic thresholding for yawning detection
                mar_history.append(mar)
                if len(mar_history) > MAR_HISTORY_LENGTH:
                    mar_history.pop(0)
                average_mar = np.mean(mar_history)
                dynamic_mar_thresh = average_mar * MAR_THRESH_MULTIPLIER

                # Yawning detection
                if mar < 0.97:  # Fixed threshold for yawning
                    YAWN_COUNTER += 1
                    if YAWN_COUNTER >= YAWN_COUNT_THRESH:
                        cv2.putText(image, "YAWNING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        yawning_detected = True
                        if DROWSY_COUNT >= DROWSY_ALERT_THRESH:
                            drowsiness_alert_counter += 1
                            DROWSY_COUNT = 0
                            print("Drowsiness and yawning detected (counter incremented).")
                            if drowsiness_alert_counter >= 3:
                                speak("Drowsiness detected, please stay alert on the road.")
                                drowsiness_alert_counter = 0
                                print("Drowsiness detected, alert spoken.")
                else:
                    YAWN_COUNTER = 0

                # Head pose distraction detection
                if abs(head_angle_x) > HEAD_POSE_THRESH or abs(head_angle_y) > HEAD_POSE_THRESH:
                    cv2.putText(image, "DISTRACTED", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    DISTRACTED_COUNT += 1
                    if DISTRACTED_COUNT >= DISTRACTED_ALERT_THRESH:
                        speak("Warning: Distracted driving detected. Please focus on the road.")
                        DISTRACTED_COUNT = 0
                        print("Distracted alert spoken.")
                else:
                    DISTRACTED_COUNT = 0

                # Displaying EAR, MAR, and head angles
                cv2.putText(image, f'EAR: {ear:.2f}', (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, f'MAR: {mar:.2f}', (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, f'Head angles: ({float(head_angle_x):.2f}, {float(head_angle_y):.2f}, {float(head_angle_z):.2f})', (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/")
def login():
    return render_template('Login.html')

@app.route('/Home.html')
def home():
    return render_template("Home.html")

@app.route('/pic_detect', methods=['GET', 'POST'])
def pic_detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            results = combined_prediction(filepath)
            print(f"Results: {results}")
            
            return render_template('pic_detect.html', results=results, uploaded_image=f'uploads/{filename}')
    return render_template('pic_detect.html', results=None)

@app.route('/vid_detect', methods=['GET', 'POST'])
def vid():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run detection on the uploaded video
            detection_results = detect_driver_distraction(filepath)

            # Create bar chart from results
            chart_path = create_bar_chart(detection_results)

            # Pass results and chart path to the template
            return render_template('vid_detect.html', results=detection_results, chart_path=chart_path)
    return render_template('vid_detect.html')

@app.route('/charts/<filename>')
def send_chart(filename):
    return send_from_directory(app.config['CHART_FOLDER'], filename)

@app.route('/about_us.html')
def about_us():
    return render_template("about_us.html")

@app.route('/live_detect.html')
def live_detect():
    return render_template("live_detect.html")

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
