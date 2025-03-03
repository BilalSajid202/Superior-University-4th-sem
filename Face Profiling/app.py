import cv2
import dlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Initialize face detector and landmark predictor
opencv_data_dir = os.path.dirname(cv2.__file__) + "/data/"
cascade_path = os.path.join(opencv_data_dir, 'D:/University/4th Sem/PAI(LAB)/Tasks/Face Profiling/haarcascade_frontalface_default.xml')

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error: Unable to load Haar Cascade classifier from {cascade_path}")
    print("Please check if the file exists and the path is correct.")
else:
    print(f"Haar Cascade classifier loaded successfully from {cascade_path}")

predictor_path = "D:/University/4th Sem/PAI(LAB)/Tasks/Face Profiling/shape_predictor_68_face_landmarks_GTX.dat"
predictor = dlib.shape_predictor(predictor_path)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return dlib.rectangle(x, y, x + w, y + h)

def get_landmarks(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face_rect)
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

def calculate_features(landmarks):
    features = {}
    face_width = np.linalg.norm(landmarks[16] - landmarks[0])
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])
    features['width_height_ratio'] = face_width / face_height if face_height != 0 else 0
    eye_distance = np.linalg.norm(landmarks[39] - landmarks[42])
    features['eye_distance_ratio'] = eye_distance / face_width if face_width != 0 else 0
    mouth_width = np.linalg.norm(landmarks[54] - landmarks[48])
    features['mouth_width_ratio'] = mouth_width / face_width if face_width != 0 else 0
    nose_length = np.linalg.norm(landmarks[33] - landmarks[27])
    features['nose_length_ratio'] = nose_length / face_height if face_height != 0 else 0
    jaw_width = np.linalg.norm(landmarks[11] - landmarks[5])
    features['jaw_width_ratio'] = jaw_width / face_height if face_height != 0 else 0
    return features

def predict_personality(features):
    personality_traits = {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5
    }
    
    if features['width_height_ratio'] > 1.9:
        personality_traits['extraversion'] += 0.1
        personality_traits['agreeableness'] -= 0.1
    
    if features['eye_distance_ratio'] > 0.25:
        personality_traits['openness'] += 0.1
        personality_traits['agreeableness'] += 0.1
    
    if features['mouth_width_ratio'] > 0.5:
        personality_traits['extraversion'] += 0.1
    
    if features['nose_length_ratio'] > 0.3:
        personality_traits['conscientiousness'] += 0.1
    
    if features['jaw_width_ratio'] > 0.8:
        personality_traits['extraversion'] += 0.1
        personality_traits['neuroticism'] -= 0.1
    
    for trait in personality_traits:
        personality_traits[trait] = max(0, min(1, personality_traits[trait]))
    
    return personality_traits

def create_radar_chart(personality_traits):
    traits = list(personality_traits.keys())
    values = list(personality_traits.values())
    num_vars = len(traits)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.1)
    plt.xticks(angles[:-1], traits)
    plt.title("Personality Traits Radar Chart")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Explicitly close the figure
    
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            # Read the image file
            image_stream = file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect face
            face_rect = detect_face(image)
            
            if face_rect is None:
                return jsonify({'error': 'No face detected in the image'})
            
            # Get landmarks and calculate features
            landmarks = get_landmarks(image, face_rect)
            features = calculate_features(landmarks)
            
            # Predict personality
            personality_traits = predict_personality(features)
            
            # Create radar chart
            radar_chart = create_radar_chart(personality_traits)
            
            return jsonify({
                'personality_traits': personality_traits,
                'radar_chart': radar_chart
            })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)