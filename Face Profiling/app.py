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

def calculate_features(landmarks, image=None, demographics=None):
    """
    Calculate facial features from landmarks with additional measurements.
    
    Parameters:
    - landmarks: Array of facial landmark points
    - image: Optional image for additional processing
    - demographics: Optional dictionary with demographic information (age, gender, ethnicity)
    
    Returns:
    - Dictionary of facial features and measurements
    """
    if landmarks is None:
        return None
    
    # Basic features from the original function
    features = {}
    
    # Face width to height ratio
    face_width = np.linalg.norm(landmarks[16] - landmarks[0])
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])
    features['width_height_ratio'] = face_width / face_height if face_height != 0 else 0
    features['face_width'] = face_width
    features['face_height'] = face_height
    
    # Eye distance to face width ratio
    eye_distance = np.linalg.norm(landmarks[39] - landmarks[42])
    features['eye_distance_ratio'] = eye_distance / face_width if face_width != 0 else 0
    
    # Mouth width to face width ratio
    mouth_width = np.linalg.norm(landmarks[54] - landmarks[48])
    features['mouth_width_ratio'] = mouth_width / face_width if face_width != 0 else 0
    features['mouth_width'] = mouth_width
    
    # Nose length to face height ratio
    nose_length = np.linalg.norm(landmarks[33] - landmarks[27])
    features['nose_length_ratio'] = nose_length / face_height if face_height != 0 else 0
    
    # Jaw width to face height ratio
    jaw_width = np.linalg.norm(landmarks[11] - landmarks[5])
    features['jaw_width_ratio'] = jaw_width / face_height if face_height != 0 else 0
    
    # NEW MEASUREMENTS
    
    # 1. Eye size ratio (eye height to face height)
    left_eye_height = np.mean([
        np.linalg.norm(landmarks[37] - landmarks[41]),
        np.linalg.norm(landmarks[38] - landmarks[40])
    ])
    right_eye_height = np.mean([
        np.linalg.norm(landmarks[43] - landmarks[47]),
        np.linalg.norm(landmarks[44] - landmarks[46])
    ])
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    features['eye_size_ratio'] = avg_eye_height / face_height if face_height != 0 else 0
    
    # 2. Lip fullness (lip height to mouth width)
    upper_lip_height = np.linalg.norm(landmarks[51] - landmarks[62])
    lower_lip_height = np.linalg.norm(landmarks[57] - landmarks[66])
    total_lip_height = upper_lip_height + lower_lip_height
    features['lip_fullness'] = total_lip_height / mouth_width if mouth_width != 0 else 0
    
    # 3. Forehead height ratio (estimate based on face height)
    # Assuming forehead starts above eyebrows (landmark 21) and ends at hairline
    # Since hairline isn't in landmarks, we'll estimate it as 30% of face height above eyebrows
    eyebrow_y = landmarks[21][1]
    estimated_hairline_y = eyebrow_y - (0.3 * face_height)
    forehead_height = eyebrow_y - estimated_hairline_y
    features['forehead_ratio'] = forehead_height / face_height if face_height != 0 else 0
    
    # 4. Chin prominence (distance from chin to mouth relative to face height)
    chin_to_mouth = np.linalg.norm(landmarks[8] - landmarks[57])
    features['chin_prominence'] = chin_to_mouth / face_height if face_height != 0 else 0
    
    # 5. Facial symmetry calculation
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    nose_tip = landmarks[33]
    
    # Measure deviation from perfect symmetry
    midline_x = nose_tip[0]
    left_deviation = abs(midline_x - left_eye_center[0])
    right_deviation = abs(right_eye_center[0] - midline_x)
    
    # Calculate symmetry score (1 = perfect symmetry, 0 = asymmetric)
    symmetry_ratio = min(left_deviation, right_deviation) / max(left_deviation, right_deviation) if max(left_deviation, right_deviation) > 0 else 1
    features['symmetry_score'] = 0.3 + (symmetry_ratio * 0.7)  # Scale to 0.3-1.0 range
    
    # Store demographic information if provided
    if demographics:
        features['demographics'] = demographics
    
    return features

def predict_personality(features):
    """
    Predict personality traits based on facial features with weighted scoring and confidence levels.
    
    Parameters:
    - features: Dictionary of facial features and measurements
    
    Returns:
    - Dictionary containing personality traits, confidence scores, and explanation
    """
    if features is None:
        # Return default values if features is None
        return {
            'personality_traits': {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            },
            'confidence': {
                'openness': 0.3,
                'conscientiousness': 0.3,
                'extraversion': 0.3,
                'agreeableness': 0.3,
                'neuroticism': 0.3
            },
            'explanation': {
                'openness': ["No significant facial features detected"],
                'conscientiousness': ["No significant facial features detected"],
                'extraversion': ["No significant facial features detected"],
                'agreeableness': ["No significant facial features detected"],
                'neuroticism': ["No significant facial features detected"]
            }
        }
    
    # Initialize personality traits with baseline values
    personality_traits = {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5
    }
    
    # Initialize confidence scores for each trait
    confidence = {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5
    }
    
    # Initialize explanation for each trait
    explanation = {
        'openness': [],
        'conscientiousness': [],
        'extraversion': [],
        'agreeableness': [],
        'neuroticism': []
    }
    
    # Define feature weights for each trait (higher = more important)
    weights = {
        'width_height_ratio': {
            'openness': 0.3,
            'conscientiousness': 0.4,
            'extraversion': 0.7,
            'agreeableness': 0.5,
            'neuroticism': 0.4
        },
        'eye_distance_ratio': {
            'openness': 0.6,
            'conscientiousness': 0.3,
            'extraversion': 0.4,
            'agreeableness': 0.6,
            'neuroticism': 0.3
        },
        'mouth_width_ratio': {
            'openness': 0.3,
            'conscientiousness': 0.3,
            'extraversion': 0.7,
            'agreeableness': 0.5,
            'neuroticism': 0.4
        },
        'nose_length_ratio': {
            'openness': 0.4,
            'conscientiousness': 0.7,
            'extraversion': 0.2,
            'agreeableness': 0.3,
            'neuroticism': 0.3
        },
        'jaw_width_ratio': {
            'openness': 0.3,
            'conscientiousness': 0.5,
            'extraversion': 0.6,
            'agreeableness': 0.4,
            'neuroticism': 0.5
        },
        'eye_size_ratio': {
            'openness': 0.6,
            'conscientiousness': 0.3,
            'extraversion': 0.4,
            'agreeableness': 0.7,
            'neuroticism': 0.4
        },
        'lip_fullness': {
            'openness': 0.4,
            'conscientiousness': 0.3,
            'extraversion': 0.6,
            'agreeableness': 0.5,
            'neuroticism': 0.3
        },
        'forehead_ratio': {
            'openness': 0.6,
            'conscientiousness': 0.5,
            'extraversion': 0.3,
            'agreeableness': 0.2,
            'neuroticism': 0.3
        },
        'chin_prominence': {
            'openness': 0.3,
            'conscientiousness': 0.6,
            'extraversion': 0.5,
            'agreeableness': 0.3,
            'neuroticism': 0.4
        },
        'symmetry_score': {
            'openness': 0.4,
            'conscientiousness': 0.5,
            'extraversion': 0.6,
            'agreeableness': 0.5,
            'neuroticism': 0.7
        }
    }
    
    # Apply demographic adjustments if available
    demographic_adjustment = 0
    if 'demographics' in features:
        demographics = features['demographics']
        
        # Age adjustments
        if 'age' in demographics:
            age = demographics['age']
            if age < 25:
                # Younger people tend to score higher on openness and neuroticism
                personality_traits['openness'] += 0.05
                personality_traits['neuroticism'] += 0.05
                explanation['openness'].append("Age factor: Younger individuals tend to score higher on openness")
                explanation['neuroticism'].append("Age factor: Younger individuals tend to score higher on neuroticism")
            elif age > 50:
                # Older people tend to score higher on agreeableness and conscientiousness
                personality_traits['agreeableness'] += 0.05
                personality_traits['conscientiousness'] += 0.05
                explanation['agreeableness'].append("Age factor: Older individuals tend to score higher on agreeableness")
                explanation['conscientiousness'].append("Age factor: Older individuals tend to score higher on conscientiousness")
        
        # Gender adjustments (based on general trends in research)
        if 'gender' in demographics:
            gender = demographics['gender']
            if gender.lower() == 'female':
                personality_traits['agreeableness'] += 0.05
                personality_traits['neuroticism'] += 0.05
                explanation['agreeableness'].append("Gender factor: Women tend to score higher on agreeableness")
                explanation['neuroticism'].append("Gender factor: Women tend to score higher on neuroticism")
            elif gender.lower() == 'male':
                personality_traits['extraversion'] += 0.03
                explanation['extraversion'].append("Gender factor: Men tend to score slightly higher on extraversion")
        
        # Ethnicity adjustments would require more nuanced research
        # This is a placeholder for potential cultural differences
        if 'ethnicity' in demographics:
            # Apply cultural adjustments if needed
            pass
    
    # Face width-to-height ratio (fWHR) - multiple thresholds with weighted effects
    face_shape = features['width_height_ratio']
    if face_shape < 1.3:  # Narrow face
        effect = 0.1 * weights['width_height_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.05
        explanation['neuroticism'].append(f"Narrow face shape (+{effect:.2f})")
        
        effect = 0.05 * weights['width_height_ratio']['openness']
        personality_traits['openness'] += effect
        confidence['openness'] += 0.05
        explanation['openness'].append(f"Narrow face shape (+{effect:.2f})")
    elif face_shape < 1.6:  # Average face
        effect = 0.05 * weights['width_height_ratio']['conscientiousness']
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += 0.05
        explanation['conscientiousness'].append(f"Average face shape (+{effect:.2f})")
        
        effect = 0.05 * weights['width_height_ratio']['agreeableness']
        personality_traits['agreeableness'] += effect
        confidence['agreeableness'] += 0.05
        explanation['agreeableness'].append(f"Average face shape (+{effect:.2f})")
    elif face_shape < 1.9:  # Wider face
        effect = 0.1 * weights['width_height_ratio']['extraversion']
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += 0.1
        explanation['extraversion'].append(f"Wider face shape (+{effect:.2f})")
    else:  # Very wide face
        effect = 0.15 * weights['width_height_ratio']['extraversion']
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += 0.15
        explanation['extraversion'].append(f"Very wide face shape (+{effect:.2f})")
        
        effect = -0.1 * weights['width_height_ratio']['agreeableness']
        personality_traits['agreeableness'] += effect
        confidence['agreeableness'] += 0.1
        explanation['agreeableness'].append(f"Very wide face shape ({effect:.2f})")
        
        effect = -0.05 * weights['width_height_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.05
        explanation['neuroticism'].append(f"Very wide face shape ({effect:.2f})")
    
    # Eye distance ratio with weighted effects
    if features['eye_distance_ratio'] < 0.2:  # Close-set eyes
        effect = 0.1 * weights['eye_distance_ratio']['conscientiousness']
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += 0.1
        explanation['conscientiousness'].append(f"Close-set eyes (+{effect:.2f})")
        
        effect = 0.05 * weights['eye_distance_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.05
        explanation['neuroticism'].append(f"Close-set eyes (+{effect:.2f})")
    elif features['eye_distance_ratio'] > 0.25:  # Wide-set eyes
        effect = 0.15 * weights['eye_distance_ratio']['openness']
        personality_traits['openness'] += effect
        confidence['openness'] += 0.15
        explanation['openness'].append(f"Wide-set eyes (+{effect:.2f})")
        
        effect = 0.1 * weights['eye_distance_ratio']['agreeableness']
        personality_traits['agreeableness'] += effect
        confidence['agreeableness'] += 0.1
        explanation['agreeableness'].append(f"Wide-set eyes (+{effect:.2f})")
    
    # Eye size ratio (new feature)
    if 'eye_size_ratio' in features:
        if features['eye_size_ratio'] > 0.05:  # Larger eyes
            effect = 0.15 * weights['eye_size_ratio']['openness']
            personality_traits['openness'] += effect
            confidence['openness'] += 0.1
            explanation['openness'].append(f"Larger eyes (+{effect:.2f})")
            
            effect = 0.1 * weights['eye_size_ratio']['agreeableness']
            personality_traits['agreeableness'] += effect
            confidence['agreeableness'] += 0.1
            explanation['agreeableness'].append(f"Larger eyes (+{effect:.2f})")
        elif features['eye_size_ratio'] < 0.03:  # Smaller eyes
            effect = 0.1 * weights['eye_size_ratio']['conscientiousness']
            personality_traits['conscientiousness'] += effect
            confidence['conscientiousness'] += 0.1
            explanation['conscientiousness'].append(f"Smaller eyes (+{effect:.2f})")
    
    # Mouth width ratio with weighted effects
    if features['mouth_width_ratio'] < 0.4:  # Narrow mouth
        effect = 0.1 * weights['mouth_width_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.1
        explanation['neuroticism'].append(f"Narrow mouth (+{effect:.2f})")
        
        effect = -0.1 * weights['mouth_width_ratio']['extraversion']
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += 0.1
        explanation['extraversion'].append(f"Narrow mouth ({effect:.2f})")
    elif features['mouth_width_ratio'] > 0.5:  # Wide mouth
        effect = 0.15 * weights['mouth_width_ratio']['extraversion']
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += 0.15
        explanation['extraversion'].append(f"Wide mouth (+{effect:.2f})")
        
        effect = 0.05 * weights['mouth_width_ratio']['agreeableness']
        personality_traits['agreeableness'] += effect
        confidence['agreeableness'] += 0.05
        explanation['agreeableness'].append(f"Wide mouth (+{effect:.2f})")
    
    # Lip fullness (new feature)
    if 'lip_fullness' in features:
        if features['lip_fullness'] > 0.4:  # Fuller lips
            effect = 0.1 * weights['lip_fullness']['extraversion']
            personality_traits['extraversion'] += effect
            confidence['extraversion'] += 0.1
            explanation['extraversion'].append(f"Fuller lips (+{effect:.2f})")
            
            effect = 0.1 * weights['lip_fullness']['openness']
            personality_traits['openness'] += effect
            confidence['openness'] += 0.1
            explanation['openness'].append(f"Fuller lips (+{effect:.2f})")
        elif features['lip_fullness'] < 0.2:  # Thinner lips
            effect = 0.1 * weights['lip_fullness']['conscientiousness']
            personality_traits['conscientiousness'] += effect
            confidence['conscientiousness'] += 0.1
            explanation['conscientiousness'].append(f"Thinner lips (+{effect:.2f})")
    
    # Nose length ratio with weighted effects
    if features['nose_length_ratio'] < 0.25:  # Short nose
        effect = 0.05 * weights['nose_length_ratio']['openness']
        personality_traits['openness'] += effect
        confidence['openness'] += 0.05
        explanation['openness'].append(f"Shorter nose (+{effect:.2f})")
    elif features['nose_length_ratio'] > 0.3:  # Long nose
        effect = 0.15 * weights['nose_length_ratio']['conscientiousness']
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += 0.15
        explanation['conscientiousness'].append(f"Longer nose (+{effect:.2f})")
        
        effect = -0.05 * weights['nose_length_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.05
        explanation['neuroticism'].append(f"Longer nose ({effect:.2f})")
    
    # Forehead ratio (new feature)
    if 'forehead_ratio' in features:
        if features['forehead_ratio'] > 0.3:  # Higher forehead
            effect = 0.15 * weights['forehead_ratio']['openness']
            personality_traits['openness'] += effect
            confidence['openness'] += 0.15
            explanation['openness'].append(f"Higher forehead (+{effect:.2f})")
            
            effect = 0.1 * weights['forehead_ratio']['conscientiousness']
            personality_traits['conscientiousness'] += effect
            confidence['conscientiousness'] += 0.1
            explanation['conscientiousness'].append(f"Higher forehead (+{effect:.2f})")
        elif features['forehead_ratio'] < 0.2:  # Lower forehead
            effect = 0.1 * weights['forehead_ratio']['extraversion']
            personality_traits['extraversion'] += effect
            confidence['extraversion'] += 0.1
            explanation['extraversion'].append(f"Lower forehead (+{effect:.2f})")
    
    # Chin prominence (new feature)
    if 'chin_prominence' in features:
        if features['chin_prominence'] > 0.2:  # Prominent chin
            effect = 0.15 * weights['chin_prominence']['conscientiousness']
            personality_traits['conscientiousness'] += effect
            confidence['conscientiousness'] += 0.15
            explanation['conscientiousness'].append(f"Prominent chin (+{effect:.2f})")
            
            effect = 0.1 * weights['chin_prominence']['extraversion']
            personality_traits['extraversion'] += effect
            confidence['extraversion'] += 0.1
            explanation['extraversion'].append(f"Prominent chin (+{effect:.2f})")
        elif features['chin_prominence'] < 0.1:  # Less prominent chin
            effect = 0.1 * weights['chin_prominence']['agreeableness']
            personality_traits['agreeableness'] += effect
            confidence['agreeableness'] += 0.1
            explanation['agreeableness'].append(f"Less prominent chin (+{effect:.2f})")
    
    # Jaw width ratio with weighted effects
    if features['jaw_width_ratio'] < 0.7:  # Narrow jaw
        effect = 0.1 * weights['jaw_width_ratio']['agreeableness']
        personality_traits['agreeableness'] += effect
        confidence['agreeableness'] += 0.1
        explanation['agreeableness'].append(f"Narrow jaw (+{effect:.2f})")
        
        effect = 0.05 * weights['jaw_width_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.05
        explanation['neuroticism'].append(f"Narrow jaw (+{effect:.2f})")
    elif features['jaw_width_ratio'] > 0.8:  # Wide jaw
        effect = 0.1 * weights['jaw_width_ratio']['extraversion']
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += 0.1
        explanation['extraversion'].append(f"Wide jaw (+{effect:.2f})")
        
        effect = -0.1 * weights['jaw_width_ratio']['neuroticism']
        personality_traits['neuroticism'] += effect
        confidence['neuroticism'] += 0.1
        explanation['neuroticism'].append(f"Wide jaw ({effect:.2f})")
        
        effect = 0.05 * weights['jaw_width_ratio']['conscientiousness']
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += 0.05
        explanation['conscientiousness'].append(f"Wide jaw (+{effect:.2f})")
    
    # Facial symmetry effects with weighted impact
    if 'symmetry_score' in features:
        symmetry = features['symmetry_score']
        symmetry_effect = (symmetry - 0.5) * 2  # Convert to -1 to 1 scale
        
        effect = symmetry_effect * 0.1 * weights['symmetry_score']['extraversion']
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += abs(symmetry_effect) * 0.1
        if effect > 0:
            explanation['extraversion'].append(f"Facial symmetry (+{effect:.2f})")
        else:
            explanation['extraversion'].append(f"Facial asymmetry ({effect:.2f})")
        
        effect = symmetry_effect * 0.1 * weights['symmetry_score']['neuroticism']
        personality_traits['neuroticism'] -= effect  # Note: inverse relationship
        confidence['neuroticism'] += abs(symmetry_effect) * 0.1
        if effect > 0:
            explanation['neuroticism'].append(f"Facial symmetry ({-effect:.2f})")
        else:
            explanation['neuroticism'].append(f"Facial asymmetry (+{-effect:.2f})")
        
        effect = symmetry_effect * 0.05 * weights['symmetry_score']['conscientiousness']
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += abs(symmetry_effect) * 0.05
        if effect > 0:
            explanation['conscientiousness'].append(f"Facial symmetry (+{effect:.2f})")
        else:
            explanation['conscientiousness'].append(f"Facial asymmetry ({effect:.2f})")
    
    # Feature combinations for more nuanced predictions
    
    # Wide face + wide jaw = leadership tendencies
    if features['width_height_ratio'] > 1.7 and features['jaw_width_ratio'] > 0.75:
        effect = 0.1 * (weights['width_height_ratio']['extraversion'] + weights['jaw_width_ratio']['extraversion']) / 2
        personality_traits['extraversion'] += effect
        confidence['extraversion'] += 0.15
        explanation['extraversion'].append(f"Wide face + wide jaw combination (+{effect:.2f})")
        
        effect = 0.1 * (weights['width_height_ratio']['conscientiousness'] + weights['jaw_width_ratio']['conscientiousness']) / 2
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += 0.1
        explanation['conscientiousness'].append(f"Wide face + wide jaw combination (+{effect:.2f})")
    
    # Large eyes + small mouth = empathetic tendencies
    if 'eye_size_ratio' in features and features['eye_size_ratio'] > 0.04 and features['mouth_width_ratio'] < 0.45:
        effect = 0.15 * (weights['eye_size_ratio']['agreeableness'] + weights['mouth_width_ratio']['agreeableness']) / 2
        personality_traits['agreeableness'] += effect
        confidence['agreeableness'] += 0.2
        explanation['agreeableness'].append(f"Large eyes + small mouth combination (+{effect:.2f})")
        
        effect = 0.1 * (weights['eye_size_ratio']['openness'] + weights['mouth_width_ratio']['openness']) / 2
        personality_traits['openness'] += effect
        confidence['openness'] += 0.15
        explanation['openness'].append(f"Large eyes + small mouth combination (+{effect:.2f})")
    
    # Long nose + narrow face = analytical tendencies
    if features['nose_length_ratio'] > 0.3 and features['width_height_ratio'] < 1.4:
        effect = 0.15 * (weights['nose_length_ratio']['conscientiousness'] + weights['width_height_ratio']['conscientiousness']) / 2
        personality_traits['conscientiousness'] += effect
        confidence['conscientiousness'] += 0.2
        explanation['conscientiousness'].append(f"Long nose + narrow face combination (+{effect:.2f})")
        
        effect = 0.05 * (weights['nose_length_ratio']['openness'] + weights['width_height_ratio']['openness']) / 2
        personality_traits['openness'] += effect
        confidence['openness'] += 0.1
        explanation['openness'].append(f"Long nose + narrow face combination (+{effect:.2f})")
    
    # Normalize scores to be between 0 and 1
    for trait in personality_traits:
        personality_traits[trait] = max(0, min(1, personality_traits[trait]))
    
    # Normalize confidence scores to be between 0 and 1
    for trait in confidence:
        confidence[trait] = max(0.1, min(0.9, confidence[trait]))
        
        # Reduce confidence if the trait score is very close to 0.5 (neutral)
        if abs(personality_traits[trait] - 0.5) < 0.1:
            confidence[trait] *= 0.8
    
    # Ensure we have explanations for all traits
    for trait in explanation:
        if not explanation[trait]:
            explanation[trait].append("No significant facial features affecting this trait")
    
    # Return the results as a dictionary
    return {
        'personality_traits': personality_traits,
        'confidence': confidence,
        'explanation': explanation
    }

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
        
        # Get demographic information if provided
        demographics = {}
        if 'age' in request.form:
            try:
                demographics['age'] = int(request.form['age'])
            except ValueError:
                pass
        
        if 'gender' in request.form:
            demographics['gender'] = request.form['gender']
        
        if 'ethnicity' in request.form:
            demographics['ethnicity'] = request.form['ethnicity']
        
        if file:
            try:
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
                
                # Pass demographics only if we have any
                if demographics and len(demographics) > 0:
                    features = calculate_features(landmarks, image, demographics)
                else:
                    features = calculate_features(landmarks)
                
                # Predict personality
                prediction_results = predict_personality(features)
                
                if prediction_results is None or 'personality_traits' not in prediction_results:
                    return jsonify({'error': 'Failed to generate personality prediction'})
                
                # Create radar chart
                radar_chart = create_radar_chart(prediction_results['personality_traits'])
                
                return jsonify({
                    'personality_traits': prediction_results['personality_traits'],
                    'confidence': prediction_results['confidence'],
                    'explanation': prediction_results['explanation'],
                    'radar_chart': radar_chart
                })
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
