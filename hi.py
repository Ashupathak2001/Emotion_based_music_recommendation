import streamlit as st
import cv2
import numpy as np
from fer import FER
import pandas as pd
from datetime import datetime
import json
import random
import time
from collections import Counter

# Enhanced emotion to music mapping with more detailed categories
EMOTION_TO_MUSIC = {
    'happy': {
        'genres': ['Pop', 'Dance', 'Upbeat Rock', 'Reggae'],
        'mood_keywords': ['uplifting', 'energetic', 'bright', 'cheerful'],
        'tempo_range': (120, 160),  # BPM (Beats Per Minute)
        'popular_artists': {
            'Pop': ['Taylor Swift', 'Ed Sheeran', 'Bruno Mars'],
            'Dance': ['Calvin Harris', 'Dua Lipa', 'The Chainsmokers'],
            'Upbeat Rock': ['Imagine Dragons', 'Maroon 5', 'The Killers'],
            'Reggae': ['Bob Marley', 'Sean Paul', 'Shaggy']
        },
        'playlist_themes': ['Summer Hits', 'Party Mix', 'Workout Energy', 'Feel Good Classics']
    },
    'sad': {
        'genres': ['Blues', 'Slow Rock', 'Classical', 'Ambient'],
        'mood_keywords': ['melancholic', 'emotional', 'deep', 'reflective'],
        'tempo_range': (60, 90),
        'popular_artists': {
            'Blues': ['B.B. King', 'Eric Clapton', 'John Lee Hooker'],
            'Slow Rock': ['Coldplay', 'The Script', 'Snow Patrol'],
            'Classical': ['Ludovico Einaudi', 'Max Richter', 'Joep Beving'],
            'Ambient': ['Brian Eno', 'Tycho', 'Jon Hopkins']
        },
        'playlist_themes': ['Rainy Day', 'Late Night Thoughts', 'Emotional Healing', 'Peaceful Piano']
    },
    'angry': {
        'genres': ['Metal', 'Punk Rock', 'Hard Rock', 'Intense Electronic'],
        'mood_keywords': ['intense', 'powerful', 'aggressive', 'energetic'],
        'tempo_range': (140, 180),
        'popular_artists': {
            'Metal': ['Metallica', 'System of a Down', 'Slipknot'],
            'Punk Rock': ['Green Day', 'Blink-182', 'Sum 41'],
            'Hard Rock': ['Foo Fighters', 'AC/DC', 'Guns N\' Roses'],
            'Intense Electronic': ['The Prodigy', 'Chemical Brothers', 'Pendulum']
        },
        'playlist_themes': ['Rage Release', 'Workout Intensity', 'Metal Classics', 'Power Hour']
    },
    'neutral': {
        'genres': ['Jazz', 'Indie', 'Folk', 'Alternative'],
        'mood_keywords': ['balanced', 'calm', 'focused', 'mindful'],
        'tempo_range': (90, 120),
        'popular_artists': {
            'Jazz': ['Miles Davis', 'John Coltrane', 'Norah Jones'],
            'Indie': ['Arctic Monkeys', 'The XX', 'Tame Impala'],
            'Folk': ['Mumford & Sons', 'The Lumineers', 'Of Monsters and Men'],
            'Alternative': ['Radiohead', 'The National', 'Bon Iver']
        },
        'playlist_themes': ['Coffee House', 'Indie Essentials', 'Focus Flow', 'Acoustic Afternoon']
    },
    'surprise': {
        'genres': ['Electronic', 'Experimental', 'Jazz Fusion', 'Progressive Rock'],
        'mood_keywords': ['unexpected', 'innovative', 'exciting', 'dynamic'],
        'tempo_range': (100, 160),
        'popular_artists': {
            'Electronic': ['Aphex Twin', 'Four Tet', 'Boards of Canada'],
            'Experimental': ['Björk', 'Flying Lotus', 'Animal Collective'],
            'Jazz Fusion': ['Weather Report', 'Snarky Puppy', 'GoGo Penguin'],
            'Progressive Rock': ['Pink Floyd', 'Tool', 'Dream Theater']
        },
        'playlist_themes': ['Mind-Bending Mix', 'Genre Fusion', 'Musical Journey', 'Discovery Weekly']
    }
}


class EnhancedMusicRecommender:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.user_preferences = self.load_user_preferences()

    def load_user_preferences(self):
        try:
            with open('user_preferences.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'favorite_genres': [],
                'disliked_genres': [],
                'preferred_tempo_range': None,
                'favorite_artists': [],
                'mood_preferences': {}
            }

    def save_user_preferences(self):
        with open('user_preferences.json', 'w') as f:
            json.dump(self.user_preferences, f)

    def get_personalized_recommendations(self, emotion):
        """Get personalized music recommendations based on emotion and user preferences"""
        if emotion not in EMOTION_TO_MUSIC:
            return None

        recommendations = {
            'current_emotion': emotion,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommendations': []
        }

        emotion_data = EMOTION_TO_MUSIC[emotion]
        preferred_genres = [genre for genre in emotion_data['genres']
                            if genre not in self.user_preferences['disliked_genres']]
        preferred_genres.extend([genre for genre in self.user_preferences['favorite_genres']
                                if genre in emotion_data['genres']])
        preferred_genres = list(set(preferred_genres))

        for genre in preferred_genres:
            if genre in emotion_data['popular_artists']:
                artists = emotion_data['popular_artists'][genre]
                artists = [artist for artist in artists if artist in self.user_preferences['favorite_artists']] or artists

                recommendations['recommendations'].append({
                    'genre': genre,
                    'artists': random.sample(artists, min(3, len(artists))),
                    'mood_keywords': random.sample(emotion_data['mood_keywords'], 2),
                    'suggested_tempo': random.randint(*emotion_data['tempo_range']),
                    'playlist_theme': random.choice(emotion_data['playlist_themes'])
                })

        return recommendations

    def analyze_major_emotion(self, cap, video_placeholder, duration=10):
        """Capture emotion data for a given duration and return the most common emotion, displaying video frames in Streamlit."""
        emotion_counts = Counter()
        end_time = time.time() + duration

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            # Display the current frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            result = self.detector.detect_emotions(frame)
            if result:
                emotions = result[0]['emotions']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                emotion_counts[dominant_emotion] += 1
            # time.sleep(0.1)  # Slight delay to avoid overloading processing

        return emotion_counts.most_common(1)[0][0] if emotion_counts else None


def create_streamlit_ui():
    st.title("Emotion-Based Music Recommender")

    recommender = EnhancedMusicRecommender()
    st.sidebar.title("User Preferences")
    
    st.sidebar.subheader("Favorite Genres")
    all_genres = set()
    for emotion_data in EMOTION_TO_MUSIC.values():
        all_genres.update(emotion_data['genres'])

    selected_genres = st.sidebar.multiselect(
        "Select your favorite genres",
        options=sorted(list(all_genres)),
        default=recommender.user_preferences['favorite_genres']
    )

    st.sidebar.subheader("Favorite Artists")
    all_artists = set()
    for emotion_data in EMOTION_TO_MUSIC.values():
        for artists in emotion_data['popular_artists'].values():
            all_artists.update(artists)

    selected_artists = st.sidebar.multiselect(
        "Select your favorite artists",
        options=sorted(list(all_artists)),
        default=recommender.user_preferences['favorite_artists']
    )

    if st.sidebar.button("Save Preferences"):
        recommender.user_preferences.update({
            'favorite_genres': selected_genres,
            'favorite_artists': selected_artists
        })
        recommender.save_user_preferences()
        st.sidebar.success("Preferences saved!")

    st.subheader("Capture Face for Emotion Analysis")
    start_analysis = st.button("Start Emotion Analysis")
    
    # Video placeholder for displaying the live video feed
    video_placeholder = st.empty()

    return recommender, start_analysis, video_placeholder


def main():
    recommender, start_analysis, video_placeholder = create_streamlit_ui()
    
    if start_analysis:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam")
            return
        
        with st.spinner("Analyzing your emotions..."):
            # Analyze the dominant emotion over a duration and show video feed
            major_emotion = recommender.analyze_major_emotion(cap, video_placeholder)
            cap.release()
        
        if major_emotion:
            recommendations = recommender.get_personalized_recommendations(major_emotion)
            
            st.subheader(f"Detected Emotion: {major_emotion.capitalize()}")
            if recommendations:
                st.write("Based on your emotion, we recommend:")
                for rec in recommendations['recommendations']:
                    st.write(f"🎵 **{rec['genre']}**")
                    st.write(f"Artists: {', '.join(rec['artists'])}")
                    st.write(f"Mood: {', '.join(rec['mood_keywords'])}")
                    st.write(f"Suggested BPM: {rec['suggested_tempo']}")
                    st.write(f"Playlist: {rec['playlist_theme']}")
                    st.write("---")
        else:
            st.write("Could not detect a consistent emotion. Please try again.")

if __name__ == "__main__":
    main()