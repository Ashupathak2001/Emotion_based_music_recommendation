import streamlit as st
import cv2
import numpy as np
from fer import FER
import tensorflow as tf

# Create a dictionary mapping emotions to music genres/moods
EMOTION_TO_MUSIC = {
    'happy': ['Pop', 'Dance', 'Upbeat Rock', 'Reggae'],
    'sad': ['Blues', 'Slow Rock', 'Classical', 'Ambient'],
    'angry': ['Metal', 'Punk Rock', 'Hard Rock', 'Intense Electronic'],
    'neutral': ['Jazz', 'Indie', 'Folk', 'Alternative'],
    'surprise': ['Electronic', 'Experimental', 'Jazz Fusion', 'Progressive Rock'],
    'fear': ['Ambient', 'Dark Classical', 'Atmospheric', 'Experimental'],
    'disgust': ['Industrial', 'Experimental', 'Dark Electronic', 'Alternative Metal']
}

# Define emotion colors for bounding boxes (BGR format)
EMOTION_COLORS = {
    'angry': (0, 0, 255),    # Red
    'sad': (255, 0, 0),      # Blue
    'neutral': (0, 255, 0),  # Green
    'happy': (0, 255, 255),  # Yellow
    'surprise': (255, 255, 0),  # Cyan
    'fear': (255, 255, 0),   # Yellow
    'disgust': (128, 0, 128) # Purple
}

class EmotionMusicRecommender:
    def __init__(self):
        # Initialize FER with MTV CLAHE preprocessing
        self.detector = FER(mtcnn=True)
        
    def preprocess_image(self, frame):
        """Preprocess image with CLAHE for better emotion detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            return gray
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return None

    def detect_emotion(self, frame):
        """Detect emotions in the given frame and return with face box coordinates"""
        try:
            # Ensure we have a valid frame
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame provided")

            # Detect emotions
            result = self.detector.detect_emotions(frame)
            
            if not result:
                return None, None, "No face detected"

            # Get the dominant emotion and box coordinates
            emotions = result[0]['emotions']
            box = result[0]['box']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            return dominant_emotion, box, None

        except Exception as e:
            return None, None, f"Error in emotion detection: {str(e)}"

    def draw_emotion_box(self, frame, box, emotion):
        """Draw a colored box around the face based on emotion"""
        if emotion in EMOTION_COLORS:
            color = EMOTION_COLORS[emotion]
            x, y, w, h = box
            
            # Draw the main rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Add emotion label with background
            label = f"{emotion.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size for background rectangle
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                        (x, y - label_height - 10), 
                        (x + label_width + 10, y), 
                        color, 
                        -1)  # Filled rectangle
            
            # Draw text
            cv2.putText(frame, 
                       label, 
                       (x + 5, y - 5), 
                       font, 
                       font_scale, 
                       (255, 255, 255),  # White text
                       thickness)
            
        return frame

    def recommend_music(self, emotion):
        """Recommend music based on detected emotion"""
        if emotion in EMOTION_TO_MUSIC:
            return EMOTION_TO_MUSIC[emotion]
        return None

def main():
    st.title("Emotion-Based Music Recommender")
    
    # Initialize the recommender
    recommender = EmotionMusicRecommender()
    
    # Create a placeholder for the webcam
    video_placeholder = st.empty()
    
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Detect emotion and get face box coordinates
            emotion, box, error = recommender.detect_emotion(frame)
            
            if error:
                st.warning(error)
                continue
                
            if emotion and box:
                # Draw emotion box on frame
                frame = recommender.draw_emotion_box(frame, box, emotion)
                
                # Get music recommendations
                recommendations = recommender.recommend_music(emotion)
                
                # Convert frame from BGR to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display results
                with video_placeholder.container():
                    # Display the frame
                    st.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Display emotion and recommendations
                    st.write(f"Detected Emotion: {emotion}")
                    if recommendations:
                        st.write("Recommended Music Genres:")
                        for genre in recommendations:
                            st.write(f"- {genre}")
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    main()