import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import math

st.title('Virtual Mouse')

st.divider()

st.subheader('Please click on `capture` to start and click on `close` to close')

with st.sidebar:
    st.divider()
    st.header('some name')
    st.divider()
    st.page_link(page='pages/sign language translator.py', label='sign language translator')
    st.page_link(page='pages/Virtual mouse.py', label='Virtual mouse')
    st.page_link(page='pages/Video translator.py', label='Video translator')


def capture():
    st.session_state.close = True

    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5)
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    min_volume = 0
    max_volume = 100

    while st.session_state.close:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                for hand_landmarks in hands:
                    if hand_landmarks and output.multi_handedness[0].classification[0].label == 'Right':

                        # Detect index and thumb fingers
                        index_finger = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_finger = hand.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        ring_finger = hand.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
                        middle_finger = hand.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

                        # Convert normalized coordinates to pixel coordinates
                        index_x = int(index_finger.x * frame_width)
                        index_y = int(index_finger.y * frame_height)
                        thumb_x = int(thumb_finger.x * frame_width)
                        thumb_y = int(thumb_finger.y * frame_height)
                        ring_x = int(ring_finger.x * frame_width)
                        ring_y = int(ring_finger.y * frame_height)
                        middle_x = int(middle_finger.x * frame_width)
                        middle_y = int(middle_finger.y * frame_height)

                        # Update cursor position
                        cursor_x = screen_width * index_finger.x
                        cursor_y = screen_height * index_finger.y
                        pyautogui.moveTo(cursor_x, cursor_y)

                        # Calculate distance between index and thumb finger
                        distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
                        scroll = math.sqrt((ring_x - thumb_x) ** 2 + (ring_y - thumb_y) ** 2)
                        neg = math.sqrt((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2)
                        # Trigger click action when index finger meets thumb finger
                        if distance < 20:  # Adjust the threshold as needed
                            pyautogui.click()
                        if scroll < 20:
                            pyautogui.scroll(60)
                        if neg < 20:
                            pyautogui.scroll(-60)

                    if hand_landmarks and output.multi_handedness[0].classification[0].label == 'Left':
                        # Detect middle and thumb fingers
                        middle_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                        thumb = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        ring_finger = hand.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]

                        # Convert normalized coordinates to pixel coordinates
                        middle_x = int(middle_finger.x * frame_width)
                        middle_y = int(middle_finger.y * frame_height)
                        thumb_x = int(thumb.x * frame_width)
                        thumb_y = int(thumb.y * frame_height)
                        ring_x = int(ring_finger.x * frame_width)
                        ring_y = int(ring_finger.y * frame_height)

                        # Calculate distance between middle finger and thumb
                        distance = math.sqrt((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2)
                        kla = math.sqrt((ring_x - thumb_x) ** 2 + (ring_y - thumb_y) ** 2)

                        # Map distance to volume range
                        volume = int((distance / frame_width) * (max_volume - min_volume))

                        # Ensure volume stays within range
                        volume = max(min(volume, max_volume), min_volume)

                        # Adjust system volume based on hand gesture
                        if distance < 30:
                            pyautogui.press('volumeup', presses=volume, interval=0.05)
                        if kla < 30:
                            pyautogui.press('volumedown', presses=volume, interval=0.05)

        cv2.imshow('VM', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


capture = st.button('capture', on_click=capture)

close = st.button('Close')

if close:
    st.session_state.close = False
