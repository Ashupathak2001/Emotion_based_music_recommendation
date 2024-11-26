# Emotion-Based YouTube Music Recommender

## 📌 Project Overview

The Emotion-Based YouTube Music Recommender is an innovative application that uses facial emotion detection to suggest personalized music videos. By analyzing your real-time emotions through webcam input, the app recommends YouTube music that matches your current mood.

## ✨ Key Features

- 🎭 Real-time emotion detection using facial recognition
- 🎵 YouTube music video recommendations based on detected emotions
- 👥 Personalized genre preferences
- 📊 Video details including views, likes, and channel information

## 🛠 Technologies Used

- Python
- Streamlit
- OpenCV (cv2)
- Facial Emotion Recognition (FER)
- YouTube Data API v3

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8+
- pip (Python package manager)

## 💾 Installation

1. Clone the repository:
```bash
git clone https://github.com/Ashupathak2001/Emotion_based_music_recommendation.git
cd emotion-music-recommender
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🔑 YouTube API Key Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the YouTube Data API v3
4. Create credentials and get an API key
5. Replace `'YOUR_API_KEY'` in the script with your actual API key

## 📦 Required Libraries

Install the necessary libraries:
```bash
pip install streamlit opencv-python fer google-api-python-client
```

## 🚀 Running the Application

```bash
streamlit run app.py
```

## 🎮 How to Use

1. Launch the application
2. Click "Start Emotion Analysis"
3. Ensure your webcam is connected
4. Sit in a well-lit area facing the camera
5. The app will detect your emotion and recommend music videos

## 🌟 User Preferences

- Customize your music recommendations by selecting:
  - Favorite music genres
  - Preferred music styles

## 🔍 Emotion Detection Methodology

The app uses advanced facial emotion recognition to detect:
- Happy
- Sad
- Angry
- Neutral
- Surprise

## 🚧 Limitations

- Requires a webcam
- Emotion detection accuracy depends on lighting and facial clarity
- Requires an active internet connection
- Limited by YouTube API quotas

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Your Name - ashupathak22@gmail.com

Project Link: [https://github.com/Ashupathak2001/Emotion_based_music_recommendation](https://github.com/Ashupathak2001/Emotion_based_music_recommendation)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [Facial Emotion Recognition](https://github.com/justinshenk/fer)
