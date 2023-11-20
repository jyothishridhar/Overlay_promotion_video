import cv2
import openpyxl
import streamlit as st
import tempfile
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

def generate_overlay_report_df(reference_overlay_frames, testing_overlay_frames):
    max_length = max(len(reference_overlay_frames), len(testing_overlay_frames))
    data = {'Reference Timestamp': [], 'Reference Frame Number': [],
            'Testing Timestamp': [], 'Testing Frame Number': [],
            'Timestamp Difference': [], 'Frame Number Difference': []}

    # Initialize variables outside the loop
    ref_timestamp, ref_frame_num = 0, 0

    for row_num in range(1, max_length + 1):
        if row_num <= len(reference_overlay_frames):
            ref_timestamp, ref_frame_num = reference_overlay_frames[row_num - 1]
            data['Reference Timestamp'].append(ref_timestamp)
            data['Reference Frame Number'].append(ref_frame_num)

        if row_num <= len(testing_overlay_frames):
            test_timestamp, test_frame_num = testing_overlay_frames[row_num - 1]
            data['Testing Timestamp'].append(test_timestamp)
            data['Testing Frame Number'].append(test_frame_num)

            # Update timestamp_diff and frame_num_diff only if reference frames are available
            if row_num <= len(reference_overlay_frames):
                timestamp_diff = test_timestamp - ref_timestamp
                frame_num_diff = test_frame_num - ref_frame_num
                data['Timestamp Difference'].append(timestamp_diff)
                data['Frame Number Difference'].append(frame_num_diff)

    df = pd.DataFrame(data)
    return df

def generate_overlay_reports(reference_overlay_frames, testing_overlay_frames, report_path):
    # Generate DataFrame
    overlay_df = generate_overlay_report_df(reference_overlay_frames, testing_overlay_frames)

    # Save to Excel
    overlay_df.to_excel(report_path, index=False)

    # Save to CSV
    csv_report_path = report_path.replace(".xlsx", ".csv")
    overlay_df.to_csv(csv_report_path, index=False)

    return overlay_df, report_path, csv_report_path

# Streamlit app code
st.title("Overlay Detection Demo")

reference_video_path = st.file_uploader("Upload Reference Video File", type=["mp4"])
testing_video_path = st.file_uploader("Upload Testing Video File", type=["mp4"])
report_path = st.text_input("Enter Report Path (e.g., overlay_report.xlsx):")

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

        reference_overlay_frames = detect_overlay(reference_path)
        testing_overlay_frames = detect_overlay(testing_path)

        overlay_df, _, _ = generate_overlay_reports(reference_overlay_frames, testing_overlay_frames, report_path)

        # Display the result on the app
        st.success("Overlay detection completed! Result:")

        # Display the DataFrame
        st.dataframe(overlay_df)

        # Provide a download link for the Excel file
        st.markdown(f"Download the result: [overlay_report.xlsx]({report_path})")
    else:
        st.warning("Please upload both reference and testing video files, and provide a report path.")
