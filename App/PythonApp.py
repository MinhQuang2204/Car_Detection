import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import threading

# Khởi tạo YOLO model
model = YOLO('../Model/best.pt')

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Video Detection")
        self.root.state('zoomed')  # Phóng to nhưng vẫn giữ taskbar
        self.video_path = None
        self.paused = True
        self.replay_flag = False
        self.current_frame = None
        self.detecting = False
        self.frame_delay = 100  # Điều chỉnh thời gian delay giữa các frame (ms)

        # Danh sách lưu các hình ảnh để tránh bị xóa bộ nhớ
        self.video_images = []  # Lưu hình ảnh video để không bị giải phóng bộ nhớ
        self.detected_images = []  # Lưu hình ảnh nhận diện để không bị giải phóng bộ nhớ

        #### Text hiển thị nội dung phía trên (vùng này màu xanh dương nhạt)
        self.label_info = tk.Label(root, text="Detect video", height=3, anchor='w', padx=10, bg="#add8e6")
        self.label_info.pack(side=tk.TOP, fill=tk.X)

        #### Tạo layout chính cho giao diện

        # Vùng bên trái chứa video (màu nền xanh lá nhạt)
        self.left_frame = tk.Frame(root, bg="#98fb98")
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Vùng bên phải hiển thị kết quả (màu nền hồng nhạt)
        self.right_frame = tk.Frame(root, bg="#ffcccb")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Vùng chiếu video, thu nhỏ một chút (màu trắng)
        self.canvas_video = tk.Canvas(self.left_frame, width=640, height=480, bg="white")
        self.canvas_video.pack()

        # Nút Play/Pause và Replay đặt dưới video (màu nền xám nhạt)
        self.control_frame = tk.Frame(self.left_frame, bg="#d3d3d3")
        self.control_frame.pack(side=tk.BOTTOM, pady=10)

        # Tải ảnh và đặt icon cho các nút
        self.play_icon = ImageTk.PhotoImage(Image.open("./img/play.png").resize((30, 30), Image.LANCZOS))
        self.pause_icon = ImageTk.PhotoImage(Image.open("./img/pause.png").resize((30, 30), Image.LANCZOS))
        self.replay_icon = ImageTk.PhotoImage(Image.open("./img/replay.png").resize((30, 30), Image.LANCZOS))
        self.select_video_icon = ImageTk.PhotoImage(Image.open("./img/select_video.png").resize((30,30), Image.LANCZOS))

        # Nút Play/Pause
        self.btn_play_pause = tk.Button(self.control_frame, image=self.play_icon, command=self.toggle_play_pause)
        self.btn_play_pause.pack(side=tk.LEFT, padx=10)
        
        # Nút Replay
        self.btn_replay = tk.Button(self.control_frame, image=self.replay_icon, command=self.replay_video)
        self.btn_replay.pack(side=tk.LEFT, padx=10)

        # Nút chọn video
        self.btn_select = tk.Button(self.control_frame, image=self.select_video_icon, command=self.load_video)
        self.btn_select.pack(side=tk.LEFT, padx=10)

        # Đường dẫn video (màu nền xám nhạt)
        self.video_path_label = tk.Label(self.control_frame, text="Hiển thị đường dẫn video", bg="#d3d3d3")
        self.video_path_label.pack(side=tk.LEFT, padx=10)

        # Nút thoát (màu nền xanh dương nhạt)
        self.btn_quit = tk.Button(self.root, text="Thoát", command=root.quit, bg="#add8e6")
        self.btn_quit.pack(side=tk.BOTTOM, pady=10)

        # Thêm dòng "Kết quả:" phía trên các khung hiển thị kết quả bên phải (màu vàng nhạt)
        self.result_label = tk.Label(self.right_frame, text="Kết quả:", font=("Arial", 14), bg="#ffffe0")
        self.result_label.pack(side=tk.TOP)

        self.video_cap = None
        self.video_playing = False

    def load_video(self):
        # Hộp thoại chọn video
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
            self.video_playing = True
            self.paused = True
            self.replay_flag = False
            self.video_path_label.config(text=self.video_path)
            self.toggle_play_pause()

    def toggle_play_pause(self):
        if self.paused:
            self.paused = False
            self.btn_play_pause.config(image=self.pause_icon)
            self.show_frame()
        else:
            self.paused = True
            self.btn_play_pause.config(image=self.play_icon)

    def show_frame(self):
        if self.video_playing and not self.paused:
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame = frame
                # Thực hiện nhận diện và hiển thị bounding box
                results = model(frame)
                frame_with_boxes = results[0].plot()  # Hiển thị bounding box lên frame

                # Chuyển đổi frame từ BGR sang RGB để hiển thị
                frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Lấy kích thước khung canvas và thay đổi kích thước ảnh video cho phù hợp
                canvas_width = self.canvas_video.winfo_width()
                canvas_height = self.canvas_video.winfo_height()
                img_resized = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img_resized)

                # Lưu hình ảnh video vào danh sách để tránh bị giải phóng bộ nhớ
                self.video_images.append(img_tk)

                # Hiển thị video trên canvas
                self.canvas_video.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.root.update()

                # Chạy YOLO nhận diện đối tượng trong một luồng riêng
                threading.Thread(target=self.run_yolo_on_frame, args=(frame,)).start()

                # Làm chậm tốc độ phát video bằng cách thêm độ trễ giữa các frame
                self.root.after(self.frame_delay, self.show_frame)
            else:
                if self.replay_flag:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.show_frame()

    def run_yolo_on_frame(self, frame):
        # Nhận diện đối tượng trên frame hiện tại
        results = model(frame)
        detected_objects = results[0].boxes  # Lấy các hộp phát hiện

        # Xóa tất cả các đối tượng detect cũ từ frame bên phải
        for widget in self.right_frame.winfo_children():
            if widget != self.result_label:  # Không xóa dòng "Kết quả:"
                widget.destroy()

        # Hiển thị các đối tượng được phát hiện trong frame
        for i, box in enumerate(detected_objects[:3]):  # Hiển thị tối đa 3 đối tượng phát hiện
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # ID lớp của đối tượng
            class_name = model.names[class_id]  # Lấy tên lớp từ ID

            detected_img = frame[y1:y2, x1:x2]  # Cắt đối tượng từ frame

            # Chuyển đổi từ BGR sang RGB để hiển thị
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            img_detected = Image.fromarray(detected_img_rgb)
            img_detected_tk = ImageTk.PhotoImage(image=img_detected)

            # Lưu hình ảnh đã cắt vào danh sách để tránh bị giải phóng bộ nhớ
            self.detected_images.append(img_detected_tk)

            # Tạo label hiển thị ảnh đã cắt ra
            label_img = tk.Label(self.right_frame, image=img_detected_tk)
            label_img.pack()

            # Hiển thị tên lớp, độ tin cậy và tọa độ bounding box
            label_info = tk.Label(self.right_frame, text=(
                f"Đối tượng {i+1}: {class_name} - Độ tin cậy: {box.conf[0]:.2f} "
                f"Tọa độ: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})"
            ), anchor='w', justify='left', bg="#ffffe0")
            label_info.pack()

    def replay_video(self):
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.replay_flag = True
            self.show_frame()

# Tạo ứng dụng tkinter
root = tk.Tk()
app = VideoPlayerApp(root)
root.mainloop()