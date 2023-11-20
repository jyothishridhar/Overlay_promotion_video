import cv2
import openpyxl
import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd

def detect_overlay(video_path):
    cap = cv2.VideoCapture(video_path)

    # Calculate histogram for the first frame
    ret, reference_frame = cap.read()
    if not ret:
        return []

    reference_hist = cv2.calcHist([reference_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    reference_hist = cv2.normalize(reference_hist, reference_hist).flatten()

    # Compare histograms of subsequent frames
    overlay_frames = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        testing_hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        testing_hist = cv2.normalize(testing_hist, testing_hist).flatten()

        # Compare histograms using correlation
        correlation = cv2.compareHist(reference_hist, testing_hist, cv2.HISTCMP_CORREL)
        if correlation < 0.9:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get timestamp in milliseconds
            overlay_frames.append((timestamp, frame_count))

    cap.release()

    return overlay_frames

def generate_overlay_excel_report(overlay_frames, report_path):
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Overlay Frames"

    sheet['A1'] = "Timestamp"
    sheet['B1'] = "Frame Number"

    for row_num, (timestamp, frame_num) in enumerate(overlay_frames, start=2):
        sheet.cell(row=row_num, column=1, value=timestamp)
        sheet.cell(row=row_num, column=2, value=frame_num)

    wb.save(report_path)

def run_overlay_detection(reference_video_path, testing_video_path, report_path, stop_flag):
    st.write("Starting overlay detection...")

    reference_overlay_frames = detect_overlay(reference_video_path)
    testing_overlay_frames = detect_overlay(testing_video_path)

    # Save the Excel report locally
    generate_overlay_excel_report(testing_overlay_frames, report_path)

    st.write("Overlay detection completed.")
    return report_path

# Streamlit app code
st.title("Overlay Detection Demo")

reference_video_path = st.file_uploader("Upload Reference Video File", type=["mp4"])
testing_video_path = st.file_uploader("Upload Testing Video File", type=["mp4"])
report_path = st.text_input("Enter Report Path (e.g., overlay_report.xlsx):")

stop_flag = [False]  # Using a list to make it mutable

if st.button("Run Overlay Detection"):
    if reference_video_path is not None and testing_video_path is not None and report_path:
        # Save the video files locally
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ref_temp, tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as test_temp:
            ref_temp.write(reference_video_path.read())
            test_temp.write(testing_video_path.read())
            reference_path = ref_temp.name
            testing_path = test_temp.name

        # Close the file handles
        ref_temp.close()
        test_temp.close()

        result_path = run_overlay_detection(reference_path, testing_path, report_path, stop_flag)

        # Display the result on the app
        st.success("Overlay detection completed! Result:")
        st.markdown(f"Download the result: [{os.path.basename(result_path)}]({result_path})")
    else:
        st.warning("Please upload both reference and testing video files, and provide a report path.")
