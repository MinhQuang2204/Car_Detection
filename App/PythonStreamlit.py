import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import time  # Thêm thư viện time để dùng time.sleep()

# Khởi tạo YOLO model
model = YOLO('yolov8n.pt')

# Thiết lập tiêu đề ứng dụng
st.title("YOLO Video Detection")

# Tải video từ người dùng
uploaded_file = st.file_uploader("Tải lên một video", type=["mp4", "avi"])

# Kiểm tra nếu có video được tải lên
if uploaded_file:
    # Đọc video
    video_path = uploaded_file.name
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    video_cap = cv2.VideoCapture(video_path)

    # Nút điều khiển video
    play_button = st.button("Play/Pause Video")
    replay_button = st.button("Replay Video")

    # Biến lưu trạng thái phát
    if "paused" not in st.session_state:
        st.session_state.paused = True

    if play_button:
        st.session_state.paused = not st.session_state.paused

    if replay_button:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.paused = False

    # Hiển thị video và thông tin nhận diện
    frame_display = st.empty()  # Vùng hiển thị frame
    results_display = st.empty()  # Vùng hiển thị kết quả

    # Phát video và xử lý từng frame
    while video_cap.isOpened():
        if not st.session_state.paused:
            ret, frame = video_cap.read()
            if not ret:
                break

            # Nhận diện đối tượng với YOLO và vẽ bounding box
            results = model(frame)
            frame_with_boxes = results[0].plot()

            # Hiển thị frame lên giao diện Streamlit
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_display.image(frame_image, caption="Video Detection", use_column_width=True)

            # Hiển thị kết quả nhận diện trong frame
            detected_objects = results[0].boxes  # Lấy các hộp phát hiện
            result_text = []
            for i, box in enumerate(detected_objects[:3]):  # Hiển thị tối đa 3 đối tượng phát hiện
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])  # ID lớp của đối tượng
                class_name = model.names[class_id]  # Lấy tên lớp từ ID
                confidence = box.conf[0]
                
                # Lưu thông tin chi tiết về đối tượng
                result_text.append(
                    f"Đối tượng {i+1}: {class_name} - Độ tin cậy: {confidence:.2f} "
                    f"Tọa độ: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})"
                )

            # Hiển thị thông tin kết quả bên dưới video
            results_display.write("Kết quả phát hiện:")
            results_display.write("\n".join(result_text))

            # Tạo khoảng thời gian nghỉ giữa các frame
            time.sleep(0.03)

        # Kết thúc video khi người dùng dừng
        else:
            st.stop()

    video_cap.release()