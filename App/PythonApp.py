### Import các thư viện cần thiết
# import thư viện đồ họa tkinter
import tkinter as tk
from tkinter import filedialog

# import thư viên pencv và PIL để phục vụ quá trình xư lý ảnh
import cv2
from PIL import Image, ImageTk

# import thư viện YOLO và paddleocr
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr

# Xử lý thread
import threading

### Khởi tạo YOLO model
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
        # Khởi tạo ocr:
        self.ocr = PaddleOCR(lang='en')

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
        self.media_path = filedialog.askopenfilename(filetypes=[("Media files", "*.gif;*.mp4;*.avi;*.mov;*.jpg;*.jpeg;*.png")])
        if self.media_path:
            if self.media_path.lower().endswith(('.gif','.mp4', '.avi', '.mov')):
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

    def update_canvas_image(self, frame):
        """Cập nhật hình ảnh lên Canvas."""
        # Chuyển đổi từ BGR sang RGB để hiển thị trên Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Tính toán tỷ lệ và vị trí để hiển thị đúng trên Canvas
        canvas_width = self.canvas_video.winfo_width()
        canvas_height = self.canvas_video.winfo_height()
        img_width, img_height = img_pil.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_resized)

        # Cập nhật ảnh trên Canvas
        if hasattr(self, 'canvas_image_id'):
            self.canvas_video.itemconfig(self.canvas_image_id, image=img_tk)
        else:
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            self.canvas_image_id = self.canvas_video.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

        # Lưu ảnh để tránh bị xóa
        self.video_images.append(img_tk)

    def get_best_detections(self, boxes, image):
        """
        Duyệt qua các bounding box và chỉ giữ lại phát hiện có độ tin cậy cao nhất cho mỗi đối tượng.
        Args:
            boxes: Các bounding box phát hiện được từ YOLO.
            image: Ảnh gốc để cắt các đối tượng phát hiện.
        Returns:
            best_detections: Từ điển chứa thông tin các phát hiện có độ tin cậy cao nhất.
        """
        best_detections = {}

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0]

            # Tạo khóa dựa trên tên lớp và vị trí trung tâm của bounding box để xác định các đối tượng tương tự
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            key = (class_name, center_x // 10, center_y // 10)  # Chia tọa độ để giảm độ nhạy

            # Kiểm tra và chỉ lưu phát hiện có độ tin cậy cao nhất cho mỗi đối tượng
            if key not in best_detections or best_detections[key]['confidence'] < confidence:
                best_detections[key] = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'coords': (x1, y1, x2, y2),
                    'image': image[y1:y2, x1:x2]
                }

        return best_detections

    def preprocess_detected_images(self, detected_boxes, image):
        """
        Preprocess các ảnh được cắt ra từ bounding box.
        Args:
            detected_boxes: Từ điển chứa thông tin bounding box đã được lọc.
            image: Ảnh gốc.
        Returns:
            processed_images: Danh sách các ảnh đã được xử lý.
        """
        processed_images = []

        for key, data in detected_boxes.items():  # Duyệt qua từng mục trong từ điển
            x1, y1, x2, y2 = data['coords']  # Truy cập tọa độ từ từ điển
            class_name = data['class_name']
            confidence = data['confidence']
            cropped_img = data['image']

            # Phóng to ảnh cắt ra lên 2.0 lần
            cropped_img_height, cropped_img_width = cropped_img.shape[:2]
            new_width = int(cropped_img_width * 2.0)
            new_height = int(cropped_img_height * 2.0)

            # Resize ảnh cắt ra
            enlarged_img = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Chuẩn hóa pixel về [0, 1] (nếu cần)
            normalized_img = enlarged_img / 255.0

            # Lưu kết quả đã xử lý
            processed_images.append({
                'class_name': class_name,
                'confidence': confidence,
                'coords': (x1, y1, x2, y2),
                'image': enlarged_img  # Ảnh đã phóng to
            })

        return processed_images

    def display_and_detect_image(self, image):
        """
        Hiển thị ảnh và phát hiện đối tượng trong ảnh đã tải.
        Args:
            image: Ảnh gốc để phát hiện đối tượng.
        """
        # Tạo bản sao của frame gốc
        original_frame = image.copy()

        # Phát hiện đối tượng và chạy OCR
        detections = self.process_frame(original_frame)

        # Vẽ bounding box, label, và OCR text lên ảnh
        processed_frame = self.draw_detections(image, detections)

        # Hiển thị các ảnh cắt ra trong khung cuộn
        self.display_detected_images(detections)

        # Cập nhật ảnh đã xử lý lên Canvas
        self.update_canvas_image(processed_frame)


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

    def process_frame(self, frame):
        """
        Phát hiện đối tượng và chạy OCR trên frame gốc.
        Args:
            frame: Frame gốc từ video.
        Returns:
            detections: Danh sách thông tin phát hiện (bounding box, class, confidence, OCR text).
        """
        results = model(frame)  # YOLO detection
        best_detections = self.get_best_detections(results[0].boxes, frame)

        # Xử lý OCR cho từng vùng được cắt
        for key, data in best_detections.items():
            cropped_img = data['image']
            ocr_results = self.ocr.ocr(cropped_img, cls=True)
            ocr_text = self.extract_ocr_text(ocr_results)  # Lấy văn bản OCR

            # Thêm văn bản OCR vào phát hiện
            best_detections[key]['ocr_text'] = ocr_text

        return best_detections
    
    def draw_detections(self, frame, detections):
        """
        Vẽ bounding box, label, và OCR text lên frame.
        Args:
            frame: Ảnh gốc.
            detections: Thông tin phát hiện từ YOLO và OCR.
        Returns:
            frame: Ảnh đã được vẽ.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        for key, data in detections.items():
            x1, y1, x2, y2 = data['coords']
            class_name = data['class_name']
            confidence = data['confidence']
            ocr_text = data.get('ocr_text', '')

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Hiển thị nhãn (class name và confidence) ngay trên bounding box
            label = f"{class_name} {confidence:.2f}"
            label_position = (x1, max(0, y1 - 10))  # Phía trên bounding box
            cv2.putText(frame, label, label_position, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

            # Hiển thị văn bản OCR ngay bên trong bounding box, dưới nhãn
            ocr_position = (x1 + 5, min(y1 + 20, y2 - 10))  # Bên trong bounding box, cách trên 20px
            cv2.putText(frame, f"OCR: {ocr_text.strip()}", ocr_position, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        return frame
    
    def display_detected_images(self, detections):
        """
        Hiển thị các ảnh đối tượng được cắt ra trong khung cuộn.
        Args:
            detections: Thông tin phát hiện đối tượng (bounding box, ảnh, OCR).
        """
        # Xóa các widget cũ trong khung cuộn
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Hiển thị các ảnh cắt ra và thông tin OCR
        self.detected_images = []
        for key, data in detections.items():
            detected_img = data['image']  # Ảnh đã cắt
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
            img_detected = Image.fromarray(detected_img_rgb)  # Chuyển sang định dạng PIL
            img_detected_tk = ImageTk.PhotoImage(image=img_detected)  # Chuyển sang định dạng Tkinter

            # Lưu tham chiếu để tránh bị xóa
            self.detected_images.append(img_detected_tk)

            # Hiển thị ảnh trong khung cuộn
            label_img = tk.Label(self.scrollable_frame, image=img_detected_tk, bg="#d3d3d3")
            label_img.image = img_detected_tk
            label_img.pack(anchor='w', padx=10, pady=5)

            # Hiển thị thông tin OCR
            ocr_text = data.get('ocr_text', '')
            label_info = tk.Label(
                self.scrollable_frame,
                text=f"Đối tượng: {data['class_name']} - Độ tin cậy: {data['confidence']:.2f}\nOCR: {ocr_text}",
                bg="#d3d3d3",
                wraplength=500,
                justify="left"
            )
            label_info.pack(anchor='w', padx=10, pady=2)
    
    def extract_ocr_text(self, ocr_results):
        """
        Trích xuất văn bản OCR từ kết quả PaddleOCR.
        Args:
            ocr_results: Kết quả OCR từ PaddleOCR.
        Returns:
            ocr_text: Văn bản đã được trích xuất (chuỗi).
        """
        ocr_text = ""
        if ocr_results:  # Kiểm tra nếu kết quả OCR không rỗng
            for line in ocr_results:
                if line:  # Kiểm tra nếu line không phải là None
                    for box in line:
                        text, conf = box[1][0], box[1][1]
                        if conf > 0.8:  # Lọc kết quả có độ tin cậy cao
                            ocr_text += text + " "
        return ocr_text.strip()  # Trả về chuỗi văn bản OCR đã được trích xuất


    def show_frame(self):
        if self.video_playing() and not self.paused:
            ret, frame = self.video_cap.read()
            if ret:
                # Tạo bản sao của frame gốc để cắt ảnh
                original_frame = frame.copy()

                # Phát hiện đối tượng và chạy OCR
                detections = self.process_frame(original_frame)  # Phát hiện đối tượng trên frame gốc

                # Cập nhật hình ảnh đã vẽ lên Canvas
                processed_frame = self.draw_detections(frame, detections)  # Vẽ bounding box, nhãn, OCR

                # Hiển thị hình ảnh lên Canvas
                self.update_canvas_image(processed_frame)

                # Hiển thị các ảnh cắt ra trong khung cuộn
                self.display_detected_images(detections)

                # Tiếp tục hiển thị frame tiếp theo sau khoảng delay
                self.root.after(self.frame_delay, self.show_frame)
            else:
                # Xử lý khi video kết thúc
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

        # Chạy OCR trên frame để phát hiện văn bản 
        ocr_results = self.ocr.ocr(frame, cls=True)  

        # Xử lý kết quả OCR và hiển thị thông tin lên khung cuộn
        for line in ocr_results:  
            for box in line: 
                text, confidence = box[1][0], box[1][1]
                if confidence > 0.8:  # Chỉ hiển thị kết quả có độ tin cậy cao
                    # Vẽ bounding box OCR và văn bản lên frame
                    cv2.rectangle(frame, tuple(box[0][0]), tuple(box[0][2]), (0, 255, 255), 2)
                    cv2.putText(frame, f"{text} ({confidence:.2f})", tuple(box[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 

                    # Hiển thị thông tin OCR trong khung cuộn
                    ocr_label_info = tk.Label(self.scrollable_frame, text=f"OCR: {text} - Độ tin cậy: {confidence:.2f}", bg="#d3d3d3")
                    ocr_label_info.pack(anchor='w', padx=10, pady=2)  

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