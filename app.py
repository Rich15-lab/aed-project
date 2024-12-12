from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

# Define tips and links for each emotion
emotion_tips = {
    "happy": {
        "advice": "You're feeling great! Celebrate your positivity!",
        "link": "https://youtu.be/dQw4w9WgXcQ"
    },
    "sad": {
        "advice": "It's okay to feel sad. Take a moment for yourself. Watch this to uplift your mood.",
        "link": "https://youtu.be/2Vv-BfVoq4g"
    },
    "angry": {
        "advice": "Take deep breaths to calm down. Consider this video to help relax.",
        "link": "https://youtu.be/5qap5aO4i9A"
    },
    "neutral": {
        "advice": "Stay grounded and keep a balanced outlook on things.",
        "link": "https://youtu.be/3jWRrafhO7M"
    },
    "fear": {
        "advice": "Courage is resistance to fear. Boost your confidence with this video.",
        "link": "https://youtu.be/KxGRhd_iWuE"
    },
    "surprise": {
        "advice": "Excited? Turn that energy into something productive! Check this out.",
        "link": "https://youtu.be/f02mOEt11OQ"
    }
}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': "No image provided. Please upload an image file to proceed."}), 400
        
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(image, actions=['emotion'])

        emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
        tip = emotion_tips.get(emotion, {"advice": "No specific advice for this emotion.", "link": ""})

        return jsonify({
            "emotion": emotion,
            "advice": tip["advice"],
            "link": tip["link"]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

