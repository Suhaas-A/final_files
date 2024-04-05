import cv2
from PIL import Image
import mediapipe as mp
import numpy as np
import joblib  # For loading the model
import time  # For timing operations
import streamlit as st
import google.generativeai as genai
import tempfile
import mimetypes

with st.sidebar:
    st.divider()
    st.header('some name')
    st.divider()
    st.page_link(page='pages/sign language translator.py', label='sign language translator')
    st.page_link(page='pages/Virtual mouse.py', label='Virtual mouse')
    st.page_link(page='pages/Video translator.py', label='Video translator')

genai.configure(api_key='AIzaSyC245Sr1xeRQGAocIgyLZr51qa9fdNOUUE')

st.title('Video Translator')

st.divider()

st.subheader('Please click on `capture` to start and click on `close` to close')


def capture(file, format):
    st.session_state.close = True

    model = joblib.load('main_files/pages/sign_language_model.pkl')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    # Initialize variables
    last_probable_letter = None
    start_time = None
    letters = []
    sign_detected = ""

    def calculate_ratios(landmarks):
        """Calculate distances between specific hand landmarks, similar to data collection phase."""
        ratios = []
        for i in range(5, 21, 4):
            for j in range(i + 1, 21, 4):
                d = np.linalg.norm(
                    np.array((landmarks[i].x, landmarks[i].y)) - np.array((landmarks[j].x, landmarks[j].y)))
                ratios.append(d)
        return np.array(ratios).reshape(1, -1)  # Reshape for a single sample

    if format == 'video':
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(''.join(letters))
                genai.configure(api_key='AIzaSyC245Sr1xeRQGAocIgyLZr51qa9fdNOUUE')
                model = genai.GenerativeModel('gemini-pro')
                letters_str = ''.join(letters)
                response = model.generate_content('Unscramble the sentence: ' + letters_str)
                unscrambled_sentence = response.text.strip()
                st.subheader(f'The sentence: {unscrambled_sentence}')
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    ratios = calculate_ratios(landmarks)

                    # Predict sign
                    prediction = model.predict(ratios)[0]  # Assuming the prediction returns an array, get the first item
                    probability = model.predict_proba(ratios).max()  # Get the highest probability

                    # Display the prediction on the frame
                    text = f"Sign: {prediction}, Probability: {probability:.2f}"
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Check if the current prediction is the same as the last one
                    if prediction == last_probable_letter:
                        if start_time is None:
                            start_time = time.time()
                        elif time.time() - start_time >= 0.5:
                            sign_detected = prediction
                            letters.append(sign_detected)
                            start_time = None
                    else:
                        start_time = None
                        last_probable_letter = prediction
                        sign_detected = ""  # Reset sign_detected when a new sign is detected

                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        pil_image = Image.open(file)
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            letter = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                ratios = calculate_ratios(landmarks)

                # Predict sign
                prediction = model.predict(ratios)[0]  # Assuming the prediction returns an array, get the first item
                probability = model.predict_proba(ratios).max()  # Get the highest probability

                # Display the prediction on the frame
                text = f"Sign: {prediction}, Probability: {probability:.2f}"
                letter.append(text)
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            st.write(letter[0])


with st.form(key='hi'):
    file = st.file_uploader('Upload the video or image file here')
    submit = st.form_submit_button('Capture')

    if submit:
        print('hi')
        mime_type, _ = mimetypes.guess_type(file.name)
        print(mime_type)
        if 'image' in mime_type:
            capture(file, 'image')
        else:
            capture(file, 'video')
