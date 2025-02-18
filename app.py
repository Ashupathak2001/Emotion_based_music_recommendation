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
from googleapiclient.discovery import build
from urllib.parse import quote
import os

# YouTube API configuration
# YOUTUBE_API_KEY = 'AIzaSyCXW8fDGgelZTD-qvbhGKrtXfDqCViugFk'  # Replace with your actual API key
youtube_api_key = 'YOUTUBE_API_KEY'
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

# Enhanced emotion to music search mapping
EMOTION_TO_MUSIC = {
    'happy': {
        'search_terms': ['upbeat music', 'happy songs', 'feel good music', 'positive vibes playlist'],
        'genres': ['pop', 'dance', 'upbeat rock'],
        'mood_keywords': ['uplifting', 'energetic', 'bright', 'cheerful'],
    },
    'sad': {
        'search_terms': ['sad songs', 'emotional music', 'melancholic playlist', 'peaceful piano'],
        'genres': ['blues', 'slow rock', 'emotional ballads'],
        'mood_keywords': ['melancholic', 'emotional', 'deep', 'reflective'],
    },
    'angry': {
        'search_terms': ['intense music', 'powerful songs', 'aggressive music', 'metal playlist'],
        'genres': ['metal', 'punk rock', 'hard rock'],
        'mood_keywords': ['intense', 'powerful', 'aggressive', 'energetic'],
    },
    'neutral': {
        'search_terms': ['relaxing music', 'calm playlist', 'indie music', 'chill songs'],
        'genres': ['indie', 'ambient', 'folk'],
        'mood_keywords': ['balanced', 'calm', 'focused', 'mindful'],
    },
    'surprise': {
        'search_terms': ['experimental music', 'unique songs', 'innovative music', 'fusion playlist'],
        'genres': ['experimental', 'fusion', 'electronic'],
        'mood_keywords': ['unexpected', 'innovative', 'exciting', 'dynamic'],
    }
}

class YouTubeMusicRecommender:
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
                'preferred_artists': []
            }

    def save_user_preferences(self):
        with open('user_preferences.json', 'w') as f:
            json.dump(self.user_preferences, f)

    def search_youtube_videos(self, search_query, max_results=5):
        """Search YouTube for music videos based on the query"""
        try:
            # Include user preferences in search query if available
            if self.user_preferences['favorite_genres']:
                genre_terms = ' OR '.join(self.user_preferences['favorite_genres'])
                search_query = f"{search_query} ({genre_terms})"

            # Perform YouTube search
            search_response = youtube.search().list(
                q=search_query,
                part='snippet',
                type='video',
                videoCategoryId='10',  # Music category
                maxResults=max_results,
                order='relevance'
            ).execute()

            videos = []
            for item in search_response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel = item['snippet']['channelTitle']
                thumbnail = item['snippet']['thumbnails']['medium']['url']
                
                # Get video statistics
                video_response = youtube.videos().list(
                    part='statistics',
                    id=video_id
                ).execute()
                
                statistics = video_response['items'][0]['statistics']
                
                videos.append({
                    'id': video_id,
                    'title': title,
                    'channel': channel,
                    'thumbnail': thumbnail,
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'views': int(statistics.get('viewCount', 0)),
                    'likes': int(statistics.get('likeCount', 0))
                })

            # Sort by views and likes
            videos.sort(key=lambda x: (x['views'], x['likes']), reverse=True)
            return videos

        except Exception as e:
            st.error(f"Error searching YouTube: {str(e)}")
            return []

    def get_recommendations(self, emotion):
        """Get YouTube music recommendations based on emotion"""
        if emotion not in EMOTION_TO_MUSIC:
            return None

        recommendations = {
            'emotion': emotion,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'videos': []
        }

        # Get emotion-based search terms
        search_terms = EMOTION_TO_MUSIC[emotion]['search_terms']
        
        # Search for videos using different search terms
        for search_term in search_terms:
            videos = self.search_youtube_videos(search_term)
            recommendations['videos'].extend(videos)

        # Remove duplicates and limit results
        seen_ids = set()
        unique_videos = []
        for video in recommendations['videos']:
            if video['id'] not in seen_ids:
                seen_ids.add(video['id'])
                unique_videos.append(video)
        
        recommendations['videos'] = unique_videos[:10]  # Limit to top 10 unique videos
        return recommendations

    def analyze_major_emotion(self, cap, video_placeholder, duration=10):
        """Capture emotion data for a given duration and return the most common emotion"""
        emotion_counts = Counter()
        end_time = time.time() + duration

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            results = self.detector.detect_emotions(frame)
            # add bounding box
            for result in results:
                box = result['box']
                x,y,w,h = box
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2) # red box

            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if results:
                emotions = results[0]['emotions']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                emotion_counts[dominant_emotion] += 1

        return emotion_counts.most_common(1)[0][0] if emotion_counts else None

def create_streamlit_ui():
    st.title("Music Recommender based on Emotions")
    
    recommender = YouTubeMusicRecommender()
    
    # Sidebar for user preferences
    st.sidebar.title("Music Preferences")
    
    # Genre preferences
    all_genres = set()
    for emotion_data in EMOTION_TO_MUSIC.values():
        all_genres.update(emotion_data['genres'])
    
    selected_genres = st.sidebar.multiselect(
        "Select your favorite genres",
        options=sorted(list(all_genres)),
        default=recommender.user_preferences['favorite_genres']
    )
    
    if st.sidebar.button("Save Preferences"):
        recommender.user_preferences.update({
            'favorite_genres': selected_genres
        })
        recommender.save_user_preferences()
        st.sidebar.success("Preferences saved!")

    # Main content
    st.subheader("Capture Face for Emotion Analysis")
    start_analysis = st.button("Start Emotion Analysis")
    video_placeholder = st.empty()

    return recommender, start_analysis, video_placeholder

def display_recommendations(recommendations):
    """Display YouTube video recommendations in Streamlit"""
    st.subheader(f"Music Recommendations based on {recommendations['emotion'].capitalize()} emotion")
    
    # Display videos in a grid layout
    cols = st.columns(2)
    for idx, video in enumerate(recommendations['videos']):
        with cols[idx % 2]:
            st.image(video['thumbnail'], use_container_width=True)
            st.markdown(f"**{video['title']}**")
            st.write(f"Channel: {video['channel']}")
            st.write(f"Views: {video['views']:,}")
            st.write(f"Likes: {video['likes']:,}")
            st.markdown(f"[Watch on YouTube]({video['url']})")
            st.write("---")

def load_custom_css(css_file_path):
    """Load and apply a custom CSS file."""
    with open(css_file_path, "r") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

def main():
    # Load and apply custom CSS
    load_custom_css("styles.css")

    recommender, start_analysis, video_placeholder = create_streamlit_ui()
    
    if start_analysis:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam")
            return
        
        with st.spinner("Analyzing your emotions..."):
            major_emotion = recommender.analyze_major_emotion(cap, video_placeholder)
            cap.release()
        
        if major_emotion:
            with st.spinner("Fetching music recommendations..."):
                recommendations = recommender.get_recommendations(major_emotion)
                if recommendations:
                    display_recommendations(recommendations)
        else:
            st.write("Could not detect a consistent emotion. Please try again.")

if __name__ == "__main__":
    main()