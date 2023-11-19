import cv2
import openpyxl
import streamlit as st
import tempfile

def detect_promotion_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Calculate histogram for the first frame
    ret, reference_frame = cap.read()
    if not ret:
        return []

    reference_hist = cv2.calcHist([reference_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    reference_hist = cv2.normalize(reference_hist, reference_hist).flatten()

    # Compare histograms of subsequent frames
    promotion_frames = []

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
            promotion_frames.append((timestamp, frame_count))

    cap.release()

    return promotion_frames


def generate_excel_report(reference_promotion_frames, testing_promotion_frames, report_path):
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Promotion Frames"

    sheet['A1'] = "Reference Timestamp"
    sheet['B1'] = "Reference Frame Number"
    sheet['C1'] = "Testing Timestamp"
    sheet['D1'] = "Testing Frame Number"
    sheet['E1'] = "Timestamp Difference"
    sheet['F1'] = "Frame Number Difference"

    max_length = max(len(reference_promotion_frames), len(testing_promotion_frames))

    for row_num in range(2, max_length + 2):
        if row_num <= len(reference_promotion_frames):
            ref_timestamp, ref_frame_num = reference_promotion_frames[row_num - 2]
            sheet.cell(row=row_num, column=1, value=ref_timestamp)
            sheet.cell(row=row_num, column=2, value=ref_frame_num)
        if row_num <= len(testing_promotion_frames):
            test_timestamp, test_frame_num = testing_promotion_frames[row_num - 2]
            sheet.cell(row=row_num, column=3, value=test_timestamp)
            sheet.cell(row=row_num, column=4, value=test_frame_num)
            if row_num <= len(reference_promotion_frames):
                timestamp_diff = test_timestamp - ref_timestamp
                frame_num_diff = test_frame_num - ref_frame_num
                sheet.cell(row=row_num, column=5, value=timestamp_diff)
                sheet.cell(row=row_num, column=6, value=frame_num_diff)

    wb.save(report_path)

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

        reference_promotion_frames = detect_promotion_video(reference_path)
        testing_promotion_frames = detect_promotion_video(testing_path)

        generate_excel_report(reference_promotion_frames, testing_promotion_frames, report_path)

        st.success("Overlay detection completed.")
    else:
        st.warning("Please upload both reference and testing video files, and provide a report path.")
