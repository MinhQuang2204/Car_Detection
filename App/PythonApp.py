########### Import các thư viện cần thiết
#### import thư viện đồ họa tkinter
import tkinter as tk
from tkinter import filedialog, PhotoImage, Label, Frame, Entry, Button, StringVar, IntVar

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

#### import thư viên opencv và PIL để phục vụ quá trình xử lý ảnh
import cv2
from PIL import Image, ImageTk

#### import thư viện YOLO và paddleocr
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr

#### Xử lý thread
import threading

#### Các thư viện khác
import os
import pandas as pd
import yaml

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
        self.frame_delay = 40  # Điều chỉnh thời gian delay giữa các frame (ms)
        self.ocr = PaddleOCR(lang='en')  # Khởi tạo OCR

        # Thuộc tính video_images lưu trữ các hình ảnh hiển thị trên Canvas
        self.video_images = []  # Khởi tạo danh sách trống
        
        # Khung chính
        self.left_frame = ttk.Frame(root, bootstyle="secondary")  # Tăng chiều rộng của left frame
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Tạo nút trên left_frame
        self.create_left_buttons()
        
        # Nội dung mặc định của right_frame
        self.init_default_content()
    
    ####################
    ###### Giao diện nút bên left frame, tạo button với các chức năng
    def create_left_buttons(self):
        """Tạo các nút bên trái để chuyển đổi tính năng."""
        button_style = {"bootstyle": "light", "width": 20, "padding": (10, 10)}  # Tăng chiều rộng và padding

        # Nút mặc định cho trang chủ
        btn_default = ttk.Button(
            self.left_frame,
            text="Trang chủ",
            command=self.init_default_content,
            **button_style
        )
        btn_default.pack(fill=tk.X, padx=10, pady=10)

        # Placeholder cho các nút tính năng khác (sẽ thêm sau)
        btn_feature_1 = ttk.Button(
            self.left_frame,
            text="Tính năng 1",
            command=self.placeholder_function,
            **button_style
        )
        btn_feature_1.pack(fill=tk.X, padx=10, pady=10)

        # Nút để chuyển đến giao diện huấn luyện mô hình
        btn_train = ttk.Button(
            self.left_frame,
            text="Train Model",
            command=self.show_train_frame,
            **button_style
        )
        btn_train.pack(fill=tk.X, padx=10, pady=10)

        # Nút "Kết quả mô hình"
        btn_results = ttk.Button(
            self.left_frame,
            text="Kết quả mô hình",
            command=self.show_model_results,
            **button_style
        )
        btn_results.pack(fill=tk.X, padx=10, pady=10)

        # Nút để hiển thị giao diện "Detect video và ảnh"
        btn_detect = ttk.Button(
            self.left_frame,
            text="Detect video và ảnh",
            command=self.show_detect_content,
            **button_style
        )
        btn_detect.pack(fill=tk.X, padx=10, pady=10)

    #######################################
    ######### nội dung nút, trang mặc định
    def init_default_content(self):
        """Khởi tạo nội dung mặc định (hiển thị thông tin)."""
        self.clear_right_frame()

        # Khung chứa icon và chữ "Trang Chủ"
        header_frame = ttk.Frame(self.right_frame, bootstyle="secondary")
        header_frame.pack(pady=(20, 10), anchor="w", padx=100)  # Căn trái toàn bộ khung

        # Icon "Trang Chủ"
        try:
            # Mở hình ảnh và thay đổi kích thước
            image = Image.open("./img/home_icon.png")
            image_resized = image.resize((30, 30), Image.LANCZOS)  # Kích thước phù hợp với chữ
            icon_image = ImageTk.PhotoImage(image_resized)

            # Hiển thị icon
            icon_label = ttk.Label(header_frame, image=icon_image, bootstyle="secondary")
            icon_label.image = icon_image  # Lưu tham chiếu để tránh bị xóa
            icon_label.pack(side="left", padx=(0, 10))  # Đặt icon bên trái và cách chữ 10px
        except Exception as e:
            print("Không thể tải hình ảnh:", e)

        # Chữ "Trang Chủ"
        home_label = ttk.Label(header_frame, text="Trang Chủ", font=("Arial", 18, "bold"), bootstyle="secondary")
        home_label.pack(side="left")

        # Tiêu đề chính
        title_label = ttk.Label(self.right_frame, text="Tiểu Luận Chuyên Ngành", font=("Arial", 24, "bold"), bootstyle="success")
        title_label.pack(pady=(20, 10))

        # Phụ đề
        subtitle_label = ttk.Label(
            self.right_frame, text="Tìm Hiểu Bài Toán Nhận Diện Biển Số Xe Qua Camera Giao Thông", font=("Arial", 16), bootstyle="secondary"
        )
        subtitle_label.pack(pady=(0, 20))

        # Tên giáo viên hướng dẫn
        advisor_label = ttk.Label(
            self.right_frame,
            text="GVHD: TS Nguyễn Thành Sơn",
            font=("Arial", 14),
            bootstyle="secondary",
            anchor="e",  # Căn phải
            justify="right",  # Chữ căn phải
        )
        advisor_label.pack(padx=60, pady=(0, 5), fill="x")  # Thêm khoảng cách lề ngang

        # Dòng "Sinh viên thực hiện:"
        student_title_label = ttk.Label(
            self.right_frame,
            text="Sinh viên thực hiện:",
            font=("Arial", 14),
            bootstyle="secondary",
            anchor="e",  # Căn phải
            justify="right",
        )
        student_title_label.pack(padx=150, pady=(0, 5), fill="x")  # Thêm khoảng cách lề ngang

        # Danh sách sinh viên
        student_label = ttk.Label(
            self.right_frame,
            text="19110457 - Phan Tấn Thành\n20110704 - Trần Minh Quang",
            font=("Arial", 14),
            bootstyle="secondary",
            anchor="e",  # Căn phải
            justify="right",  # Chữ căn phải
        )
        student_label.pack(padx=50, pady=(0, 20), fill="x")  # Thêm khoảng cách lề ngang

        # Footer
        footer_label = ttk.Label(self.right_frame, text="Made with Tkinter", font=("Arial", 10, "italic"), bootstyle="secondary")
        footer_label.pack(side="bottom", pady=(0, 10))

    ###########################
    #### Tạo các button với chức năng
      
    # def create_left_buttons(self):
    #     # Nút mặc định cho trang chủ
    #     btn_default = tk.Button(self.left_frame, text="Trang chủ", command=self.init_default_content, bg="#ffffff")
    #     btn_default.pack(fill=tk.X, padx=10, pady=5)

    #     # Placeholder cho các nút tính năng khác (sẽ thêm sau)
    #     btn_feature_1 = tk.Button(self.left_frame, text="Tính năng 1", command=self.placeholder_function, bg="#ffffff")
    #     btn_feature_1.pack(fill=tk.X, padx=10, pady=5)

    #     # Nút để chuyển đến giao diện huấn luyện mô hình
    #     btn_train = tk.Button(self.left_frame, text="Train Model", command=self.show_train_frame, bg="#ffffff")
    #     btn_train.pack(fill=tk.X, padx=10, pady=5)

    #     # Nút để hiển thị giao diện "Detect video và ảnh"
    #     btn_detect = tk.Button(self.left_frame, text="Detect video và ảnh", command=self.show_detect_content, bg="#ffffff")
    #     btn_detect.pack(fill=tk.X, padx=10, pady=5)

    ##################################
    #### Phần mô phỏng quá trình training với tùy chỉnh tham số 

    def show_train_frame(self):
        """Hiển thị giao diện để huấn luyện mô hình."""
        self.clear_right_frame()

        ttk.Label(self.right_frame, text="Train Model", font=("Arial", 18, "bold"), bootstyle="danger").pack(pady=20)

        # Đường dẫn đến tập dữ liệu train
        ttk.Label(self.right_frame, text="Đường dẫn đến dữ liệu training:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.train_data_path = StringVar()
        train_entry = ttk.Entry(self.right_frame, textvariable=self.train_data_path, width=100, state="readonly")  # Đặt state="readonly"
        train_entry.pack(anchor="w", padx=20, pady=5)
        ttk.Button(self.right_frame, text="Chọn", command=self.select_train_path, bootstyle="primary").pack(anchor="w", padx=20, pady=5)

        # Đường dẫn đến dữ liệu validation
        ttk.Label(self.right_frame, text="Đường dẫn đến dữ liệu validation:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.val_data_path = StringVar()
        val_entry = ttk.Entry(self.right_frame, textvariable=self.val_data_path, width=100, state="readonly")  # Đặt state="readonly"
        val_entry.pack(anchor="w", padx=20, pady=5)
        ttk.Button(self.right_frame, text="Chọn", command=self.select_val_path, bootstyle="primary").pack(anchor="w", padx=20, pady=5)

        # Đường dẫn đến model
        ttk.Label(self.right_frame, text="Đường dẫn đến mô hình:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.model_path = StringVar()
        model_entry = ttk.Entry(self.right_frame, textvariable=self.model_path, width=100, state="readonly")  # Đặt state="readonly"
        model_entry.pack(anchor="w", padx=20, pady=5)
        ttk.Button(self.right_frame, text="Chọn", command=self.select_model_path, bootstyle="primary").pack(anchor="w", padx=20, pady=5)

        # Batch size
        ttk.Label(self.right_frame, text="Chọn giá trị Batch size:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.batch_size = IntVar(value=16)
        batch_size_slider = ttk.Scale(self.right_frame, from_=1, to=128, orient="horizontal", variable=self.batch_size, length=500, bootstyle="danger")
        batch_size_slider.pack(anchor="w", padx=20, pady=5)

        # Hiển thị giá trị của batch size
        self.batch_size_label = ttk.Label(self.right_frame, text=f"Batch size: {self.batch_size.get()}", bootstyle="secondary")
        self.batch_size_label.pack(anchor="w", padx=20, pady=5)
        batch_size_slider.bind("<Motion>", self.update_batch_size_label)

        # Epochs
        ttk.Label(self.right_frame, text="Chọn giá trị Số epochs:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.epochs = IntVar(value=120)
        epochs_slider = ttk.Scale(self.right_frame, from_=1, to=200, orient="horizontal", variable=self.epochs, length=500, bootstyle="success")
        epochs_slider.pack(anchor="w", padx=20, pady=5)

        # Hiển thị giá trị của epochs
        self.epochs_label = ttk.Label(self.right_frame, text=f"Số epochs: {self.epochs.get()}", bootstyle="secondary")
        self.epochs_label.pack(anchor="w", padx=20, pady=5)
        epochs_slider.bind("<Motion>", self.update_epochs_label)

        # Nút bắt đầu train
        ttk.Button(
            self.right_frame,
            text="Bắt đầu",
            command=self.start_training,
            bootstyle="success-outline",
            width=20,  # Tăng độ rộng của nút
        ).pack(anchor="w", padx=20, pady=20)  # Căn trái và thêm khoảng cách

    def update_batch_size_label(self, event):
        """Cập nhật nhãn hiển thị giá trị batch size."""
        self.batch_size_label.config(text=f"Batch size: {self.batch_size.get()}")

    def update_epochs_label(self, event):
        """Cập nhật nhãn hiển thị giá trị epochs."""
        self.epochs_label.config(text=f"Số epochs: {self.epochs.get()}")

    def select_train_path(self):
        """Chọn đường dẫn dữ liệu training."""
        path = filedialog.askdirectory()
        if path:
            self.train_data_path.set(path)  # Cập nhật giá trị vào hộp nhập

    def select_val_path(self):
        """Chọn đường dẫn dữ liệu validation."""
        path = filedialog.askdirectory()
        if path:
            self.val_data_path.set(path)  # Cập nhật giá trị vào hộp nhập

    def select_model_path(self):
        """Chọn đường dẫn mô hình."""
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt;*.pth")])
        if path:
            self.model_path.set(path)  # Cập nhật giá trị vào hộp nhập

    def start_training(self):
        """Hàm xử lý khi nhấn nút bắt đầu train."""
        train_path = self.train_data_path.get()
        val_path = self.val_data_path.get()
        model_path = self.model_path.get()
        batch_size = self.batch_size.get()
        epochs = self.epochs.get()

        if not train_path or not val_path or not model_path:
            tk.messagebox.showerror("Lỗi", "Hãy chọn đầy đủ các đường dẫn cần thiết.")
            return

        # Hiển thị thông tin cấu hình
        tk.messagebox.showinfo("Thông tin", f"Đang huấn luyện mô hình từ:\n"
                                            f"Training path: {train_path}\n"
                                            f"Validation path: {val_path}\n"
                                            f"Model path: {model_path}\n"
                                            f"Batch size: {batch_size}\n"
                                            f"Epochs: {epochs}")

    ###################################
    ##### Phần hiển thị các thông số đánh giá của mô hình
    def show_model_results(self):
        """Hiển thị giao diện chọn thư mục và kết quả mô hình."""
        self.clear_right_frame()

        # Tạo khung chọn thư mục
        self.selection_frame = ttk.Frame(self.right_frame,)
        self.selection_frame.pack(fill=tk.X, padx=20, pady=10)

        # Tiêu đề
        ttk.Label(self.selection_frame, text="Kết quả mô hình", font=("Arial", 18, "bold"), bootstyle="danger").pack(pady=20)
        
        # Chọn đường dẫn thư mục
        ttk.Label(self.selection_frame, text="Chọn thư mục chứa mô hình:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.model_dir_path = StringVar()
        model_dir_entry = ttk.Entry(self.selection_frame, textvariable=self.model_dir_path, width=150, state="readonly")
        model_dir_entry.pack(anchor="w", padx=20, pady=5)
        ttk.Button(self.selection_frame, text="Chọn", command=self.select_model_dir, bootstyle="primary").pack(anchor="w", padx=20, pady=5)

        # Tạo khung hiển thị kết quả
        self.result_frame = ttk.Frame(self.right_frame,)
        self.result_frame.pack(fill=tk.X, padx=20, pady=10)

    def clear_results(self):
        """Xóa nội dung hiển thị kết quả trong result_frame."""
        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def select_model_dir(self):
        """Chọn thư mục chứa mô hình và hiển thị kết quả nếu tìm được file."""
        base_path = filedialog.askdirectory()
        if base_path:
            # Ghép thêm /runs/detect/train
            model_dir = os.path.join(base_path, "runs", "detect", "train")

            # Kiểm tra thư mục có tồn tại không
            if not os.path.exists(model_dir):
                self.clear_results()
                ttk.Label(self.result_frame, text="Thư mục không tồn tại.", bootstyle="danger").pack(pady=10)
                return

            # Cập nhật đường dẫn và tính toán kết quả
            self.model_dir_path.set(model_dir)
            self.calculate_results()

    def calculate_results(self):
        """Tính toán kết quả từ các file trong thư mục và hiển thị."""
        model_dir = self.model_dir_path.get()

        # Tìm file cần thiết
        results_file = os.path.join(model_dir, 'results.csv')
        args_file = os.path.join(model_dir, 'args.yaml')
        best_model_file = os.path.join(model_dir, 'weights', 'best.pt')

        if not os.path.exists(results_file) or not os.path.exists(args_file):
            self.clear_results()
            ttk.Label(self.result_frame, text="Không tìm thấy file results.csv hoặc args.yaml.", bootstyle="danger").pack(pady=10)
            return

        # Đọc file results.csv
        try:
            data = pd.read_csv(results_file)
        except Exception as e:
            self.clear_results()
            ttk.Label(self.result_frame, text=f"Lỗi khi đọc file results.csv: {e}", bootstyle="danger").pack(pady=10)
            return

        # Tính toán train_loss, val_loss, object_accuracy
        data['train_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        data['val_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        data['object_accuracy'] = 2 * (data['metrics/precision(B)'] * data['metrics/recall(B)']) / (
            data['metrics/precision(B)'] + data['metrics/recall(B)']
        )
        data['object_accuracy'] = data['object_accuracy'].fillna(0)  # Xử lý chia cho 0

        # Tính fitness
        data['fitness'] = (0.1 * data['metrics/mAP50(B)'] + 0.9 * data['metrics/mAP50-95(B)'])
        best_epoch = data.loc[data['fitness'].idxmax()]

        # Chỉ lấy thời gian tại dòng tốt nhất
        best_time_seconds = best_epoch['time']
        hours, remainder = divmod(best_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"

        # Đọc args.yaml
        try:
            with open(args_file, 'r') as file:
                args_data = yaml.safe_load(file)
            batch_size = args_data.get('batch', 'Không xác định')
        except Exception as e:
            self.clear_results()
            ttk.Label(self.result_frame, text=f"Lỗi khi đọc file args.yaml: {e}", bootstyle="danger").pack(pady=10)
            return

        # Xóa kết quả cũ trước khi hiển thị kết quả mới
        self.clear_results()

        # Hiển thị kết quả trong result_frame
        ttk.Label(self.result_frame, text="Kết quả mô hình sau khi huấn luyện:", font=("Arial", 18, "bold"), bootstyle="success").pack(anchor="w", pady=10)
        result_text = (
            #f"Đường dẫn mô hình: {model_dir}\n"
            f"Số Epochs: {int(best_epoch['epoch'])}\n"
            f"Thời gian huấn luyện tại epoch tốt nhất: {formatted_time}\n"
            f"Train Loss: {best_epoch['train_loss']:.4f}\n"
            f"Validation Loss: {best_epoch['val_loss']:.4f}\n"
            f"Object Accuracy: {best_epoch['object_accuracy']:.4f}\n"
            f"mAP@0.5: {best_epoch['metrics/mAP50(B)']:.4f}\n"
            f"mAP@0.5:0.95: {best_epoch['metrics/mAP50-95(B)']:.4f}\n"
            f"Kích thước batch: {batch_size}"
        )
        ttk.Label(self.result_frame, text=result_text, font=('Arial', 14), justify="left", bootstyle="secondary").pack(anchor="w", pady=10)

    ##################################
    ###### Xóa nội dung right frame khi bấm sang button mới bên khung left frame
    def clear_right_frame(self):
        """Xóa toàn bộ nội dung trong khung phải."""
        for widget in self.right_frame.winfo_children():
            widget.destroy()

    def placeholder_function(self):
        """Placeholder cho các nút tính năng khác."""
        self.clear_right_frame()
        label = tk.Label(self.right_frame, text="Tính năng này chưa được triển khai.", bg="#d3d3d3")
        label.pack(expand=True)

    ########################################
    ####### Phần detect video/ảnh hay demo 
    def show_detect_content(self):
        """Hiển thị nội dung hiện tại của Detect video và ảnh."""
        self.clear_right_frame()
        
        # Khung chứa nút tìm kiếm và đường dẫn video
        self.top_controls = tk.Frame(self.right_frame, bg="#d3d3d3")
        self.top_controls.pack(side=tk.TOP, anchor='w', pady=10)

        # Nút chọn video
        self.select_video_icon = ImageTk.PhotoImage(Image.open("./img/select_video.png").resize((30, 30), Image.LANCZOS))
        self.btn_select = tk.Button(self.top_controls, image=self.select_video_icon, command=self.load_media)
        self.btn_select.pack(side=tk.LEFT, padx=10)

        # Label hiển thị đường dẫn video
        self.video_path_label = ttk.Label(self.top_controls, text="No video selected", bootstyle="secondary-inverse", anchor="w")  # Đặt màu nền secondary
        self.video_path_label.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        # Canvas phát video
        self.canvas_video = tk.Canvas(self.right_frame, width=720, height=360, bg="white")
        self.canvas_video.pack(side=tk.TOP, pady=10)

        # Nút Play/Pause và Replay
        self.control_frame = tk.Frame(self.right_frame, bg="#d3d3d3")
        self.control_frame.pack(side=tk.TOP, pady=10)

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