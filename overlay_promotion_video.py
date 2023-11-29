import cv2
import streamlit as st
import tempfile
import pandas as pd
import streamlit.components.v1 as components

def install_gdown():
    components.html(
        """
        <h2>Installing gdown...</h2>
        <p>This may take a minute.</p>
        """
    )
    st.experimental_memo(gdown.install)
    components.html("<h2>gdown has been installed!</h2>")

# Try to import gdown, install it if not available
try:
    import gdown
except ImportError:
    st.warning("gdown is not installed. Installing now...")
    st.warning("This might take a minute.")

    # Install gdown
    st.code("!pip install gdown")
    import gdown  # Try to import again

    # Check if the import was successful
    try:
        import gdown
        st.success("gdown has been successfully installed.")
    except ImportError:
        st.error("Failed to install gdown. Please check the logs for more information.")

# Continue with the rest of your code...


def download_google_drive_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


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
        else:
            # If no reference frame, append NaN or 0 to maintain the array length
            data['Reference Timestamp'].append(0)
            data['Reference Frame Number'].append(0)

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
            else:
                # If no reference frame, append NaN or 0 to maintain the array length
                data['Timestamp Difference'].append(0)
                data['Frame Number Difference'].append(0)

    df = pd.DataFrame(data)
    return df

def generate_overlay_reports(reference_overlay_frames, testing_overlay_frames):
    # Generate DataFrame
    overlay_df = generate_overlay_report_df(reference_overlay_frames, testing_overlay_frames)

    # Save to CSV
    csv_report_path = tempfile.mktemp(suffix=".csv")
    overlay_df.to_csv(csv_report_path, index=False)

    return overlay_df, csv_report_path

# Streamlit app code
st.title("Overlay Detection Demo")

# Replace these file uploader lines with text input for Google Drive file IDs
reference_video_id = st.text_input("Enter Reference Video Google Drive File ID:")
testing_video_id = st.text_input("Enter Testing Video Google Drive File ID:")

if st.button("Run Overlay Detection"):
    if reference_video_id != "" and testing_video_id != "":
        # Save the video files locally
        reference_path = tempfile.mktemp(suffix=".mp4")
        testing_path = tempfile.mktemp(suffix=".mp4")

        # Download videos from Google Drive
        download_google_drive_file(reference_video_id, reference_path)
        download_google_drive_file(testing_video_id, testing_path)

        reference_overlay_frames = detect_overlay(reference_path)
        testing_overlay_frames = detect_overlay(testing_path)

        overlay_df, _ = generate_overlay_reports(reference_overlay_frames, testing_overlay_frames)

        # Display the result on the app
        st.success("Overlay detection completed! Result:")

        # Display the DataFrame
        st.dataframe(overlay_df)

    else:
        st.warning("Please enter both reference and testing video Google Drive File IDs.")
