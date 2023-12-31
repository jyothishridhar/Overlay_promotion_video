import cv2
import streamlit as st
import tempfile
import requests
import io
import pandas as pd
import os

def download_video(url):
    response = requests.get(url)
    return io.BytesIO(response.content)

def detect_overlay(video_content):
    # Save video content to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(video_content.getvalue())
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)

    overlay_frames = []

    try:
        ret, reference_frame = cap.read()

        # Calculate histogram for the first frame
        reference_hist = cv2.calcHist([reference_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        reference_hist = cv2.normalize(reference_hist, reference_hist).flatten()

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
                # Get the timestamp of the current frame
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                overlay_frames.append((timestamp, cap.get(cv2.CAP_PROP_POS_FRAMES)))

            print(f"Correlation: {correlation}")

    finally:
        cap.release()
        # Clean up the temporary file
        os.remove(temp_file_path)

    return overlay_frames

def generate_overlay_report_df(reference_overlay_frames, testing_overlay_frames):
    max_length = max(len(reference_overlay_frames), len(testing_overlay_frames))
    data = {'Reference Timestamp(ms)': [], 'Reference Frame Number': [],
            'Testing Timestamp(ms)': [], 'Testing Frame Number': [],
            'Timestamp Difference': [], 'Frame Number Difference': [], 'Promotion Video Status': []}

    # Initialize variables outside the loop
    ref_timestamp, ref_frame_num = 0, 0

    for row_num in range(1, max_length + 1):
        if row_num <= len(reference_overlay_frames):
            ref_timestamp, ref_frame_num = reference_overlay_frames[row_num - 1]
            data['Reference Timestamp(ms)'].append(ref_timestamp)
            data['Reference Frame Number'].append(ref_frame_num)
        else:
            # If no reference frame, append NaN or 0 to maintain the array length
            data['Reference Timestamp(ms)'].append(0)
            data['Reference Frame Number'].append(0)

        if row_num <= len(testing_overlay_frames):
            test_timestamp, test_frame_num = testing_overlay_frames[row_num - 1]
            data['Testing Timestamp(ms)'].append(test_timestamp)
            data['Testing Frame Number'].append(test_frame_num)

            # Update timestamp_diff and frame_num_diff only if reference frames are available
            if row_num <= len(reference_overlay_frames):
                timestamp_diff = test_timestamp - ref_timestamp
                frame_num_diff = test_frame_num - ref_frame_num
                data['Timestamp Difference'].append(timestamp_diff)
                data['Frame Number Difference'].append(frame_num_diff)

                # Determine Promotion Video Status
                if timestamp_diff == 0 and frame_num_diff == 0:
                    data['Promotion Video Status'].append('Same Frame')
                else:
                    data['Promotion Video Status'].append('Overlay detected')
            else:
                # If no reference frame, append NaN or 0 to maintain the array length
                data['Timestamp Difference'].append(0)
                data['Frame Number Difference'].append(0)
                data['Promotion Video Status'].append('')

    df = pd.DataFrame(data)
    return df

def generate_overlay_reports(reference_overlay_frames, testing_overlay_frames):
    # Print the length of overlay frames
    print("Reference Overlay Frames:", len(reference_overlay_frames))
    print("Testing Overlay Frames:", len(testing_overlay_frames))

    # Generate DataFrame
    overlay_df = generate_overlay_report_df(reference_overlay_frames, testing_overlay_frames)

    # Save to CSV
    csv_report_path = tempfile.mktemp(suffix=".csv")
    overlay_df.to_csv(csv_report_path, index=False)

    return overlay_df, csv_report_path

# Streamlit app code
st.title("Overlay Detection Demo")

# Git LFS URLs for the videos
reference_video_url = "https://github.com/jyothishridhar/Overlay_promotion_video/raw/main/concat_video_1.mp4"
testing_video_url = "https://github.com/jyothishridhar/Overlay_promotion_video/raw/main/concat_video_2.mp4"

# Download videos
reference_video_content = download_video(reference_video_url)
testing_video_content = download_video(testing_video_url)

# Add download links
st.markdown(f"**Download Reference Video**")
st.markdown(f"[Click here to download the Reference Video]({reference_video_url})")

st.markdown(f"**Download Testing Video**")
st.markdown(f"[Click here to download the Testing Video]({testing_video_url})")

if st.button("Run Overlay Detection"):
    reference_overlay_frames = detect_overlay(reference_video_content)
    testing_overlay_frames = detect_overlay(testing_video_content)

    overlay_df, csv_report_path = generate_overlay_reports(reference_overlay_frames, testing_overlay_frames)

    # Display the result on the app
    st.success("Overlay detection completed! Result:")

    # Display the DataFrame
    st.dataframe(overlay_df)

    # Add download link for the report
    st.markdown(f"**Download Overlay Report**")
    st.markdown(f"[Click here to download the Overlay Report CSV]({csv_report_path})")
