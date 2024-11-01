import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

# Khởi tạo YOLO model
model = YOLO('../Model/Models_yolov10n_dataBienSoNhieuLoaiv4_datanoscale_anhmau1nhan/runs/detect/train/weights/best.pt')

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv10 App Video Detection")
        self.root.state('zoomed')  # Phóng to nhưng vẫn giữ taskbar
        self.video_path = None
        self.paused = True
        self.replay_flag = False
        self.current_frame = None
        self.frame_delay = 100  # Điều chỉnh thời gian delay giữa các frame (ms)
        
        # Khung chính
        self.left_frame = tk.Frame(root, width=300, bg="#d3d3d3")  # Màu xám nhạt
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        self.right_frame = tk.Frame(root, bg="#d3d3d3")  # Màu xám nhạt
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Khung chứa nút tìm kiếm và đường dẫn video
        self.top_controls = tk.Frame(self.right_frame, bg="#d3d3d3")  # Màu xám nhạt
        self.top_controls.pack(side=tk.TOP, anchor='w', pady=10)

        # Sửa lại nút chọn media để dùng cho cả ảnh và video
        self.select_video_icon = ImageTk.PhotoImage(Image.open("./img/select_video.png").resize((30, 30), Image.LANCZOS))
        self.btn_select = tk.Button(self.top_controls, image=self.select_video_icon, command=self.load_media)
        self.btn_select.pack(side=tk.LEFT, padx=10)

        # Label hiển thị đường dẫn video bên cạnh nút Find
        self.video_path_label = tk.Label(self.top_controls, text="No video selected", bg="#d3d3d3", anchor='w')
        self.video_path_label.pack(side=tk.LEFT, padx=10)

        # Khung video (Canvas) để phát video
        self.canvas_video = tk.Canvas(self.right_frame, width=720, height=360, bg="white")
        self.canvas_video.pack(side=tk.TOP, pady=10)

        # Khung chứa nút Play/Pause và Replay
        self.control_frame = tk.Frame(self.right_frame, bg="#d3d3d3")  # Màu xám nhạt
        self.control_frame.pack(side=tk.TOP, pady=10)
        
        # Nút Play/Pause và Replay
        self.play_icon = ImageTk.PhotoImage(Image.open("./img/play.png").resize((30, 30), Image.LANCZOS))
        self.pause_icon = ImageTk.PhotoImage(Image.open("./img/pause.png").resize((30, 30), Image.LANCZOS))
        self.replay_icon = ImageTk.PhotoImage(Image.open("./img/replay.png").resize((30, 30), Image.LANCZOS))

        self.btn_play_pause = tk.Button(self.control_frame, image=self.play_icon, command=self.toggle_play_pause)
        self.btn_play_pause.pack(side=tk.LEFT, padx=10)

        self.btn_replay = tk.Button(self.control_frame, image=self.replay_icon, command=self.replay_video)
        self.btn_replay.pack(side=tk.LEFT, padx=10)

        # Khung hiển thị kết quả với thanh cuộn
        self.result_frame = tk.Frame(self.right_frame, bg="#d3d3d3")
        self.result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        self.canvas_results = tk.Canvas(self.result_frame, bg="#d3d3d3")
        self.scrollbar = tk.Scrollbar(self.result_frame, orient="vertical", command=self.canvas_results.yview)
        self.scrollable_frame = tk.Frame(self.canvas_results, bg="#d3d3d3")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_results.configure(scrollregion=self.canvas_results.bbox("all"))
        )

        self.canvas_results.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_results.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_results.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.video_cap = None
        self.video_images = []
        self.detected_images = []

    def load_media(self):
        # Hộp thoại chọn media (ảnh hoặc video)
        self.media_path = filedialog.askopenfilename(filetypes=[("Media files", "*.mp4;*.avi;*.mov;*.jpg;*.jpeg;*.png")])
        if self.media_path:
            if self.media_path.lower().endswith(('.mp4', '.avi', '.mov')):
                # Nếu là video
                self.video_path = self.media_path
                self.video_cap = cv2.VideoCapture(self.video_path)
                self.video_path_label.config(text=self.video_path)  # Cập nhật đường dẫn video vào Label
                self.paused = True
                self.replay_flag = False
                self.toggle_play_pause()
            elif self.media_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Nếu là ảnh
                image = cv2.imread(self.media_path)
                self.display_and_detect_image(image)

    def display_and_detect_image(self, image):
        # Chạy YOLO để phát hiện đối tượng
        results = model(image)
        detected_objects = results[0].boxes

        # Dùng font OpenCV cho bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Xóa các widget cũ trong khung cuộn trước khi hiển thị kết quả mới
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Vẽ bounding box và văn bản cho mỗi đối tượng được phát hiện
        for i, box in enumerate(detected_objects):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0]

            # Vẽ bounding box màu xanh lá cây và văn bản lên ảnh
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Hiển thị thông tin đối tượng trong khung cuộn
            detected_img = image[y1:y2, x1:x2]
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            img_detected = Image.fromarray(detected_img_rgb)
            img_detected_tk = ImageTk.PhotoImage(image=img_detected)

            label_img = tk.Label(self.scrollable_frame, image=img_detected_tk, bg="#d3d3d3")
            label_img.image = img_detected_tk  # Lưu ảnh để tránh bị xóa
            label_img.pack(anchor='w', padx=10, pady=5)

            label_info = tk.Label(self.scrollable_frame, text=(
                f"Đối tượng {i+1}: {class_name} - Độ tin cậy: {confidence:.2f} "
                f"Tọa độ: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})"
            ), anchor='w', justify='left', bg="#d3d3d3")
            label_info.pack(anchor='w', padx=10, pady=2)

        # Chuyển đổi từ BGR sang RGB để hiển thị trên Tkinter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)

        # Resize ảnh để vừa với canvas
        canvas_width = self.canvas_video.winfo_width()
        canvas_height = self.canvas_video.winfo_height()
        img_width, img_height = img_pil.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_resized)

        # Hiển thị ảnh lên canvas
        self.canvas_video.delete("all")
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        self.canvas_video.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
        self.video_images.append(img_tk)  # Lưu giữ ảnh để tránh bị xóa

    def run_yolo_on_image(self, image):
        results = model(image)
        detected_objects = results[0].boxes

        # Xóa các widget cũ trong khung cuộn trước khi hiển thị kết quả mới
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Hiển thị các đối tượng được phát hiện trong khung cuộn
        for i, box in enumerate(detected_objects[:3]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detected_img = image[y1:y2, x1:x2]
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            img_detected = Image.fromarray(detected_img_rgb)
            img_detected_tk = ImageTk.PhotoImage(image=img_detected)

            self.detected_images.append(img_detected_tk)
            label_img = tk.Label(self.scrollable_frame, image=img_detected_tk, bg="#d3d3d3")
            label_img.pack(anchor='w', padx=10, pady=5)

            label_info = tk.Label(self.scrollable_frame, text=(
                f"Đối tượng {i+1}: {class_name} - Độ tin cậy: {box.conf[0]:.2f} "
                f"Tọa độ: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})"
            ), anchor='w', justify='left', bg="#d3d3d3")
            label_info.pack(anchor='w', padx=10, pady=2)

    def toggle_play_pause(self):
        if self.paused:
            self.paused = False
            self.btn_play_pause.config(image=self.pause_icon)
            self.show_frame()
        else:
            self.paused = True
            self.btn_play_pause.config(image=self.play_icon)

    def show_frame(self):
        if self.video_playing() and not self.paused:
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame = frame
                results = model(frame)  # Phát hiện đối tượng trên frame

                # Dùng font OpenCV cho bounding box
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Xóa các widget cũ trong khung cuộn trước khi hiển thị kết quả mới
                for widget in self.scrollable_frame.winfo_children():
                    widget.destroy()

                # Vẽ bounding box và văn bản cho mỗi đối tượng được phát hiện
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = box.conf[0]

                    # Vẽ bounding box màu xanh lá cây và văn bản lên frame
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Hiển thị thông tin đối tượng trong khung cuộn
                    detected_img = frame[y1:y2, x1:x2]
                    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                    img_detected = Image.fromarray(detected_img_rgb)
                    img_detected_tk = ImageTk.PhotoImage(image=img_detected)

                    label_img = tk.Label(self.scrollable_frame, image=img_detected_tk, bg="#d3d3d3")
                    label_img.image = img_detected_tk  # Lưu ảnh để tránh bị xóa
                    label_img.pack(anchor='w', padx=10, pady=5)

                    label_info = tk.Label(self.scrollable_frame, text=(
                        f"Đối tượng {i+1}: {class_name} - Độ tin cậy: {confidence:.2f} "
                        f"Tọa độ: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})"
                    ), anchor='w', justify='left', bg="#d3d3d3")
                    label_info.pack(anchor='w', padx=10, pady=2)

                # Chuyển đổi frame từ BGR sang RGB để hiển thị
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Tính toán tỷ lệ để đảm bảo video nằm trong khung mà không bị tràn
                canvas_width = self.canvas_video.winfo_width()
                canvas_height = self.canvas_video.winfo_height()
                frame_width, frame_height = img.size
                scale = min(canvas_width / frame_width, canvas_height / frame_height)
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img_resized)

                self.video_images.append(img_tk)
                self.canvas_video.delete("all")
                x_offset = (canvas_width - new_width) // 2
                y_offset = (canvas_height - new_height) // 2
                self.canvas_video.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
                self.root.update()

                # Tạo thread để tiếp tục xử lý frame tiếp theo
                self.root.after(self.frame_delay, self.show_frame)
            else:
                # Nếu hết video và replay_flag được đặt, phát lại video từ đầu
                if self.replay_flag:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.show_frame()

    def run_yolo_on_frame(self, frame):
        results = model(frame)
        detected_objects = results[0].boxes

        # Xóa các widget cũ trong khung cuộn trước khi hiển thị kết quả mới
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Hiển thị các đối tượng được phát hiện trong khung cuộn
        for i, box in enumerate(detected_objects[:3]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detected_img = frame[y1:y2, x1:x2]
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            img_detected = Image.fromarray(detected_img_rgb)
            img_detected_tk = ImageTk.PhotoImage(image=img_detected)

            self.detected_images.append(img_detected_tk)
            label_img = tk.Label(self.scrollable_frame, image=img_detected_tk, bg="#d3d3d3")
            label_img.pack(anchor='w', padx=10, pady=5)

            label_info = tk.Label(self.scrollable_frame, text=(
                f"Đối tượng {i+1}: {class_name} - Độ tin cậy: {box.conf[0]:.2f} "
                f"Tọa độ: (x1: {x1}, y1: {y1}), (x2: {x2}, y2: {y2})"
            ), anchor='w', justify='left', bg="#d3d3d3")
            label_info.pack(anchor='w', padx=10, pady=2)

    def replay_video(self):
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.replay_flag = True
            self.show_frame()

    def video_playing(self):
        return self.video_cap is not None and self.video_cap.isOpened()
    

# Tạo ứng dụng tkinter
root = tk.Tk()
app = VideoPlayerApp(root)
root.mainloop()