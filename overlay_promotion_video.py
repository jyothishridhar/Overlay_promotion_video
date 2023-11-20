import cv2
from openpyxl import Workbook
import streamlit as st
import os
import numpy as np
import tempfile
import pandas as pd

def run_logo_detection(logo_path, video_path, stop_flag):
    st.write("Starting logo detection...")

    # Read the logo image from the file uploader
    logo_bytes = logo_path.read()
    logo_np = np.frombuffer(logo_bytes, np.uint8)
    logo = cv2.imdecode(logo_np, cv2.IMREAD_COLOR)
    gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_logo, descriptors_logo = sift.detectAndCompute(gray_logo, None)

    # Save the video file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_path.read())
        video_path = temp_video.name

    # Open the locally saved video file
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    # Lists to store frame numbers and detection status
    frame_numbers = []
    detection_statuses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1  # Increment the frame number

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize the SIFT detector and compute the keypoints and descriptors for the frame
        keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

        # Match the descriptors between the logo and the frame
        matcher = cv2.BFMatcher()
        matches = matcher.match(descriptors_logo, descriptors_frame)

        # Sort the matches by their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Display the frame number
        st.write(f"Frame: {frame_number}")

        # If enough good matches are found, consider the logo is detected
        if len(matches) > 10:  # Adjust the threshold value as per your requirement
            logo_detected = True
            detection_status = 'Logo Detected'
        else:
            logo_detected = False
            detection_status = 'Logo Not Detected'

        # Append frame number and detection status to the lists
        frame_numbers.append(frame_number)
        detection_statuses.append(detection_status)

        # Display the frame with logo detection status and frame number
        st.image(frame, channels="BGR")
        st.write(f'Detection Status: {detection_status}, Frame: {frame_number}')

        # Check the stop_flag to stop detection
        if stop_flag[0]:
            break

    # Save the Excel workbook
    result_df = pd.DataFrame({'Frame Number': frame_numbers, 'Logo Detection Status': detection_statuses})
    result_path = os.path.join(os.getcwd(), "logo_detection_report.xlsx")
    result_df.to_excel(result_path, index=False)

    # Clean up the temporary video file
    os.unlink(video_path)

    st.write("Logo detection completed.")
    return result_df

# Streamlit app code
st.title("Logo Detection Demo")

logo_path = st.file_uploader("Upload Logo Image", type=["png", "jpg", "jpeg"])
video_path = st.file_uploader("Upload Video File", type=["mp4"])

stop_flag = [False]  # Using a list to make it mutable

if st.button("Run Demo"):
    if logo_path is not None and video_path is not None:
        result_df = run_logo_detection(logo_path, video_path, stop_flag)

        # Display the result on the app
        st.success("Demo completed! Result:")

        # Display the DataFrame
        st.dataframe(result_df)

    else:
        st.warning("Please upload both the logo and video files.")
