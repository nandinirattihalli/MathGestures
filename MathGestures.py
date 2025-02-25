import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import time

st.set_page_config(layout="wide")
st.image('MathGestures_Transparent.png')

col1, col2 = st.columns([3, 2])
with col1:
    start_camera, stop_camera = st.columns(2)
    with start_camera:
        start_camera = st.button("📸 Turn Camera On")
    with stop_camera:
        stop_camera = st.button("🔴 Turn Camera Off")
    run = st.toggle('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")
    response_time_area = st.subheader("")
    warning_area = st.subheader("")

# Adding an Interesting Sidebar Section
st.sidebar.title("🎨 Math with Gestures")
st.sidebar.image('MathGestures_Transparent.png', width=200)
st.sidebar.markdown("---")
st.sidebar.header("📌 How to Use")
st.sidebar.markdown("1️⃣ *Click 'Enable Camera'* to start the recognition.")
st.sidebar.markdown("2️⃣ *Use gestures:* ✌ (Draw), ✊ (Erase), 🖐 (Solve)")
st.sidebar.markdown("3️⃣ *View AI Results* in the right panel.")
st.sidebar.markdown("4️⃣ *Click 'Disable Camera'* to stop.")
st.sidebar.markdown("---")
st.sidebar.header("✨ Features")
st.sidebar.markdown("🔹 AI-powered gesture recognition")
st.sidebar.markdown("🔹 Real-time math problem solving")
st.sidebar.markdown("🔹 Interactive and user-friendly interface")
st.sidebar.markdown("---")
st.sidebar.header("📧 Contact Us")
st.sidebar.markdown("💡 Have feedback? Reach us at: [support@example.com](mailto:nandinirattihalli4@gmail.com)")

# Configure AI model
genai.configure(api_key="AIzaSyDA7g14xMQs8cqqehj-NxOeiKpj2a6iY_0")
model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"max_output_tokens": 1000, "temperature": 0.5})

cap = None
prev_pos = None
canvas = None
output_text = ""
response_time = None

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def is_poor_lighting(canvas, threshold=50):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    non_black_pixels = np.count_nonzero(gray > 10)
    return non_black_pixels < threshold

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 1]:
        if is_poor_lighting(canvas):
            return "⚠ Poor lighting detected! Improve lighting and try again.", None
        pil_image = Image.fromarray(canvas).convert('L').resize((128, 128))
        start_time = time.time()
        response = model.generate_content(["Solve this math problem", pil_image])
        response_time = time.time() - start_time
        return response.text, response_time
    return None, None

if start_camera:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(5, 720)
    detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.7, minTrackCon=0.5)

while cap and cap.isOpened() and run:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        response, response_time = sendToAI(model, canvas, fingers)
        if response:
            output_text = response
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    if output_text:
        output_text_area.text(output_text)
    if response_time:
        response_time_area.text(f"AI Response Time: {response_time:.2f} seconds")
    if is_poor_lighting(canvas):
        warning_area.text("⚠ Poor lighting detected! Improve lighting and try again.")
    else:
        warning_area.text("")
    cv2.waitKey(1)