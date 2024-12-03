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

# #### Xử lý thread
import threading
import multiprocessing

#### Các thư viện khác
import os
import pandas as pd
import yaml
import shutil
import random

# Thư viện xử lý toán học và mảng
import numpy as np
import torch

# Thư viện hỗ trợ các phép tính và đọc dữ liệu
from torchvision.ops import box_iou

### Khởi tạo YOLO model
#model = YOLO('../Model/Models_yolov10n_dataBienSoNhieuLoaiv4_datanoscale_anhmau1nhan/runs/detect/train/weights/best.pt')

def train_yolov10(stop_event, data_yaml_path, output_dir, batch_size, epochs):
    """Huấn luyện mô hình YOLOv10."""
    try:
        model = YOLO('yolov10n.pt')  # Tải mô hình YOLOv10
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            patience=10,
            project=output_dir,
        )
        if stop_event.is_set():
            print("Quá trình huấn luyện YOLOv10 đã bị dừng.")
    except Exception as e:
        print(f"Lỗi khi huấn luyện YOLOv10: {e}")


def train_other_model(stop_event, data_yaml_path, output_dir, batch_size, epochs):
    """Huấn luyện cho các mô hình khác (placeholder)."""
    try:
        print("Huấn luyện mô hình khác...")
        # Placeholder cho logic huấn luyện các mô hình khác
        if stop_event.is_set():
            print("Quá trình huấn luyện mô hình khác đã bị dừng.")
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình khác: {e}")


class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần Mềm Nhận Diện Biển Số Xe Qua Video")
        self.root.state('zoomed')  # Phóng to nhưng vẫn giữ taskbar
        self.video_path = None
        self.paused = True
        self.replay_flag = False
        self.current_frame = None
        self.frame_delay = 15  # Điều chỉnh thời gian delay giữa các frame (ms)
        self.ocr = None #PaddleOCR(lang='en')  # Khởi tạo OCR

        self.train_process = None  # Tiến trình huấn luyện
        self.stop_event = multiprocessing.Event()  # Sự kiện dừng tiến trình

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

        # Nút để chuyển đến giao diện test mô hình
        btn_test = ttk.Button(
            self.left_frame,
            text="Test Model",
            command=self.show_test_frame,
            **button_style
        )
        btn_test.pack(fill=tk.X, padx=10, pady=10)

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

    #####################################
    #####################################
    #### Phần mô phỏng quá trình training với tùy chỉnh tham số 
    ################################

    def show_train_frame(self):
        """Hiển thị giao diện để huấn luyện mô hình."""
        self.clear_right_frame()

        ttk.Label(self.right_frame, text="Train Model", font=("Arial", 18, "bold"), bootstyle="danger").pack(pady=20)

        # Đường dẫn đến tập dữ liệu train
        ttk.Label(self.right_frame, text="Đường dẫn đến dữ liệu:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.train_data_path = StringVar()
        train_entry = ttk.Entry(self.right_frame, textvariable=self.train_data_path, width=100, state="readonly")  # Đặt state="readonly"
        train_entry.pack(anchor="w", padx=20, pady=5)
        ttk.Button(self.right_frame, text="Chọn", command=self.select_train_path, bootstyle="primary").pack(anchor="w", padx=20, pady=5)

        # Tỷ lệ Train/Test
        ttk.Label(self.right_frame, text="Tỷ lệ Train/Test (%):", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.train_test_ratio = IntVar(value=85)  # Mặc định 85% dữ liệu cho Train
        train_test_slider = ttk.Scale(
            self.right_frame,
            from_=50,
            to=95,
            orient="horizontal",
            variable=self.train_test_ratio,
            length=500,
            bootstyle="danger",
            command=self.update_train_test_label  # Liên kết cập nhật trực tiếp
        )
        train_test_slider.pack(anchor="w", padx=20, pady=5)

        # Hiển thị tỷ lệ Train/Test
        self.train_test_label = ttk.Label(
            self.right_frame,
            text=f"Tỷ lệ Train: {self.train_test_ratio.get()}% - Test: {100 - self.train_test_ratio.get()}%",
            bootstyle="secondary",
        )
        self.train_test_label.pack(anchor="w", padx=20, pady=5)

        # Tỷ lệ Train/Valid
        ttk.Label(self.right_frame, text="Tỷ lệ Train/Valid (%):", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.train_valid_ratio = IntVar(value=80)  # Giá trị mặc định là Train = 80%
        ratio_slider = ttk.Scale(
            self.right_frame,
            from_=50,  # Tối thiểu Train = 50%
            to=95,  # Tối đa Train = 95%
            orient="horizontal",
            variable=self.train_valid_ratio,
            length=500,
            bootstyle="danger",
            command=self.update_ratio_label  # Liên kết hàm cập nhật nhãn
        )
        ratio_slider.pack(anchor="w", padx=20, pady=5)

        # Hiển thị tỷ lệ Train/Valid
        self.ratio_label = ttk.Label(
            self.right_frame,
            text=f"Tỷ lệ Train: {self.train_valid_ratio.get()}% - Valid: {100 - self.train_valid_ratio.get()}%",
            bootstyle="secondary",
        )
        self.ratio_label.pack(anchor="w", padx=20, pady=5)

        # ComboBox chọn mô hình
        ttk.Label(self.right_frame, text="Chọn mô hình:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.selected_model = StringVar(value="Chọn mô hình")
        self.model_combo = ttk.Combobox(
            self.right_frame,
            textvariable=self.selected_model,
            values=["YOLOv10",],  # Danh sách mô hình YOLO
            state="readonly",
            width=50,
        )
        self.model_combo.pack(anchor="w", padx=20, pady=5)

        # Nút điều khiển (Bắt đầu/Dừng)
        self.control_button = ttk.Button(
            self.right_frame,
            text="Bắt đầu",
            command=self.toggle_train_process,
            bootstyle="success-outline",
            width=20,
        )
        self.control_button.pack(anchor="w", padx=20, pady=20)

    def update_train_test_label(self, event=None):
        """Cập nhật nhãn hiển thị tỷ lệ Train/Test."""
        train_ratio = self.train_test_ratio.get()
        test_ratio = 100 - train_ratio
        self.train_test_label.config(text=f"Tỷ lệ Train: {train_ratio}% - Test: {test_ratio}%")

    def update_ratio_label(self, event=None):
        """Cập nhật nhãn hiển thị tỷ lệ Train/Valid."""
        train_ratio = self.train_valid_ratio.get()  # Giá trị từ thanh kéo (Train)
        valid_ratio = 100 - train_ratio  # Tính tỷ lệ Valid
        self.ratio_label.config(text=f"Tỷ lệ Train: {train_ratio}% - Valid: {valid_ratio}%")

    def select_train_path(self):
        """Chọn đường dẫn dữ liệu training và thiết lập tỉ lệ ban đầu."""
        path = filedialog.askdirectory()
        if path:
            self.train_data_path.set(path)  # Cập nhật giá trị vào hộp nhập

            try:
                structure_type, train_count, valid_count, test_count = self.check_dataset_structure(path)
            
                # Tính tỉ lệ ban đầu
                if structure_type == "YOLOv8":
                    total_count = train_count + valid_count + test_count
                    self.initial_train_test_ratio = int(((train_count + valid_count) / total_count) * 100)
                    self.initial_train_valid_ratio = int((train_count / (train_count + valid_count)) * 100) if valid_count > 0 else 100
                else:  # UNSPLIT
                    self.initial_train_test_ratio = 85  # Mặc định 85:15
                    self.initial_train_valid_ratio = 80  # Mặc định 80:20

                # Cập nhật tỉ lệ vào giao diện
                self.train_test_ratio.set(self.initial_train_test_ratio)
                self.train_valid_ratio.set(self.initial_train_valid_ratio)
                self.update_train_test_label()
                self.update_ratio_label()
            except ValueError as e:
                tk.messagebox.showerror("Lỗi", str(e))

    def toggle_train_process(self):
        """Chuyển đổi giữa trạng thái Bắt đầu và Dừng huấn luyện."""
        if not hasattr(self, "is_training"):
            self.is_training = False

        if self.is_training:
            # Nếu đang huấn luyện, dừng tiến trình
            self.stop_training()
        else:
            # Nếu chưa huấn luyện, bắt đầu huấn luyện
            self.start_training()

    def check_dataset_structure(self, dataset_path):
        """
        Kiểm tra cấu trúc dataset và xác định số lượng ảnh trong mỗi tập.
        """
        train_images_path = os.path.join(dataset_path, "train", "images")
        valid_images_path = os.path.join(dataset_path, "valid", "images")
        test_images_path = os.path.join(dataset_path, "test", "images")

        train_count = len(os.listdir(train_images_path)) if os.path.exists(train_images_path) else 0
        valid_count = len(os.listdir(valid_images_path)) if os.path.exists(valid_images_path) else 0
        test_count = len(os.listdir(test_images_path)) if os.path.exists(test_images_path) else 0

        if train_count + valid_count + test_count > 0:
            return "YOLOv8", train_count, valid_count, test_count

        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        if os.path.exists(images_path) and os.path.exists(labels_path):
            total_count = len(os.listdir(images_path))
            return "UNSPLIT", total_count, 0, 0

        raise ValueError("Dataset không hợp lệ hoặc thiếu cấu trúc cần thiết.")
        
    def calculate_ratios(self, train_count, valid_count, test_count):
        """
        Tính toán các tỷ lệ Train/Test, Valid, và Train.
        Args:
            train_count: Số lượng mẫu trong tập Train.
            valid_count: Số lượng mẫu trong tập Valid.
            test_count: Số lượng mẫu trong tập Test.
        Returns:
            train_test_ratio: Tỷ lệ (Train + Valid) trên tổng.
            valid_ratio: Tỷ lệ Valid trên (Train + Valid).
            train_ratio: Tỷ lệ Train trên (Train + Valid).
        """
        total_count = train_count + valid_count + test_count
        # Tính tỷ lệ Train/Test
        train_test_ratio = int(((train_count + valid_count) / total_count) * 100) if total_count > 0 else 0
        # Tính tỷ lệ Valid
        valid_ratio = int((valid_count / (train_count + valid_count)) * 100) if (train_count + valid_count) > 0 else 0
        # Tính tỷ lệ Train
        train_ratio = 100 - valid_ratio  # Train + Valid luôn bằng 100%

        return train_test_ratio, valid_ratio, train_ratio

    # hàm chia dataset theo tỉ lệ train và tỉ lệ valid
    def split_dataset(self, dataset_path, train_ratio, valid_ratio):
        """Chia dataset thành train, valid, test dựa trên tỷ lệ cung cấp."""
        # Kiểm tra tỷ lệ
        if train_ratio <= 0 or train_ratio > 100:
            raise ValueError("Tỷ lệ Train không hợp lệ. Vui lòng chọn giá trị từ 1 đến 100.")
        if valid_ratio < 0 or valid_ratio >= 100:
            raise ValueError("Tỷ lệ Valid không hợp lệ. Vui lòng chọn giá trị từ 0 đến 99.")

        # Tạo thư mục sao lưu tạm thời
        temp_backup_path = os.path.join(dataset_path, "temp_backup")
        temp_images_path = os.path.join(temp_backup_path, "images")
        temp_labels_path = os.path.join(temp_backup_path, "labels")

        os.makedirs(temp_images_path, exist_ok=True)
        os.makedirs(temp_labels_path, exist_ok=True)

        # Kiểm tra cấu trúc dataset
        train_images_path = os.path.join(dataset_path, "train", "images")
        train_labels_path = os.path.join(dataset_path, "train", "labels")
        valid_images_path = os.path.join(dataset_path, "valid", "images")
        valid_labels_path = os.path.join(dataset_path, "valid", "labels")
        test_images_path = os.path.join(dataset_path, "test", "images")
        test_labels_path = os.path.join(dataset_path, "test", "labels")

        all_pairs = []  # Danh sách tổng hợp ảnh
        dataset_structure = None

        # Phát hiện cấu trúc
        if os.path.exists(train_images_path) or os.path.exists(valid_images_path) or os.path.exists(test_images_path):
            dataset_structure = "YOLOv8"
            print("Phát hiện cấu trúc YOLOv8")
            for path in [train_images_path, valid_images_path, test_images_path]:
                if os.path.exists(path):
                    image_files = sorted(os.listdir(path))
                    for image in image_files:
                        label = image.replace(".jpg", ".txt")
                        label_path = os.path.join(path.replace("images", "labels"), label)
                        if os.path.exists(label_path):
                            all_pairs.append((os.path.join(path, image), label_path))
        elif os.path.exists(os.path.join(dataset_path, "images")) and os.path.exists(os.path.join(dataset_path, "labels")):
            dataset_structure = "UNSPLIT"
            print("Phát hiện cấu trúc UNSPLIT")
            images_path = os.path.join(dataset_path, "images")
            labels_path = os.path.join(dataset_path, "labels")
            image_files = sorted(os.listdir(images_path))
            for image in image_files:
                label = image.replace(".jpg", ".txt")
                label_path = os.path.join(labels_path, label)
                if os.path.exists(label_path):
                    all_pairs.append((os.path.join(images_path, image), label_path))
        else:
            raise ValueError("Dataset không hợp lệ. Vui lòng kiểm tra cấu trúc thư mục.")

        # Kiểm tra danh sách cặp ảnh-nhãn
        if not all_pairs:
            raise ValueError("Không tìm thấy bất kỳ cặp ảnh-nhãn hợp lệ nào trong dataset.")

        # Sao chép dữ liệu vào `temp_backup`
        for image_path, label_path in all_pairs:
            shutil.copy(image_path, os.path.join(temp_images_path, os.path.basename(image_path)))
            shutil.copy(label_path, os.path.join(temp_labels_path, os.path.basename(label_path)))

        # Trộn và chia dataset
        random.shuffle(all_pairs)
        total_count = len(all_pairs)
        train_count = int((train_ratio / 100) * total_count)
        valid_count = int((valid_ratio / 100) * train_count)
        valid_count = min(valid_count, train_count)
        test_count = total_count - train_count

        train_pairs = all_pairs[:train_count]
        valid_pairs = train_pairs[-valid_count:]
        train_pairs = train_pairs[:-valid_count]
        test_pairs = all_pairs[train_count:]

        print(f"Tổng số mẫu: {total_count}")
        print(f"Số mẫu Train: {train_count}")
        print(f"Số mẫu Valid: {valid_count}")
        print(f"Số mẫu Test: {test_count}")

        # Sao chép cặp ảnh vào thư mục đích
        def copy_files(pair_list, dest_images_path, dest_labels_path):
            for image_path, label_path in pair_list:
                os.makedirs(dest_images_path, exist_ok=True)
                os.makedirs(dest_labels_path, exist_ok=True)
                shutil.copy(image_path, os.path.join(dest_images_path, os.path.basename(image_path)))
                shutil.copy(label_path, os.path.join(dest_labels_path, os.path.basename(label_path)))

        copy_files(train_pairs, train_images_path, train_labels_path)
        copy_files(valid_pairs, valid_images_path, valid_labels_path)
        copy_files(test_pairs, test_images_path, test_labels_path)

        # Cập nhật file data.yaml
        self.update_data_yaml(dataset_path, train_images_path, valid_images_path, test_images_path)

        # Xóa thư mục UNSPLIT
        if dataset_structure == "UNSPLIT":
            shutil.rmtree(os.path.join(dataset_path, "images"), ignore_errors=True)
            shutil.rmtree(os.path.join(dataset_path, "labels"), ignore_errors=True)
            print("Đã xóa các thư mục UNSPLIT.")

        # Xóa thư mục tạm
        shutil.rmtree(temp_backup_path)
        print("Đã xóa thư mục tạm.")


    def update_data_yaml(self, dataset_path, train_images_path, valid_images_path, test_images_path):
        """
        Cập nhật nội dung file data.yaml, chỉ update các đường dẫn.
        Args:
            dataset_path: Đường dẫn gốc của dataset.
            train_images_path: Đường dẫn đến tập train/images.
            valid_images_path: Đường dẫn đến tập valid/images.
            test_images_path: Đường dẫn đến tập test/images.
        """
        # Đường dẫn file data.yaml
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        # Nếu file `data.yaml` đã tồn tại, chỉ update các đường dẫn
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, "r") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
        else:
            # Nếu file chưa tồn tại, tạo mới cấu trúc yaml cơ bản
            yaml_data = {
                "nc": 1,  # Số lượng class (mặc định là 1, cập nhật nếu cần)
                "names": ["bienso"]  # Tên class (mặc định là "bienso")
            }
        # Cập nhật đường dẫn mới
        yaml_data["train"] = os.path.abspath(train_images_path)
        yaml_data["val"] = os.path.abspath(valid_images_path)
        yaml_data["test"] = os.path.abspath(test_images_path)
        # Ghi lại file `data.yaml`
        with open(data_yaml_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        print(f"File data.yaml đã được cập nhật: {data_yaml_path}")

    def start_training(self):
        """Khởi động tiến trình huấn luyện."""
        train_path = self.train_data_path.get()
        selected_model = self.selected_model.get()
        current_train_test_ratio = self.train_test_ratio.get()
        current_train_valid_ratio = self.train_valid_ratio.get()

        if not train_path or selected_model == "Chọn mô hình":
            tk.messagebox.showerror("Lỗi", "Hãy chọn mô hình và thông tin đầy đủ.")
            return

        # Kiểm tra cấu trúc dataset
        try:
            structure_type, train_count, valid_count, test_count = self.check_dataset_structure(train_path)
        except ValueError as e:
            tk.messagebox.showerror("Lỗi", str(e))
            return

        # Kiểm tra và thực hiện chia lại dataset
        if (
            structure_type == "UNSPLIT" or
            current_train_test_ratio != self.initial_train_test_ratio or
            current_train_valid_ratio != self.initial_train_valid_ratio
        ):
            # Tính lại tỷ lệ train/valid/test
            train_ratio = current_train_test_ratio
            valid_ratio = int(train_ratio * (100 - current_train_valid_ratio) / 100)

            print("Tỷ lệ mới:")
            print(f"Train Ratio: {train_ratio}%")
            print(f"Valid Ratio: {valid_ratio}%")
            print(f"Test Ratio: {100 - train_ratio}%")

            # Chia lại dataset
            self.split_dataset(train_path, train_ratio, valid_ratio)

            # Cập nhật lại tỷ lệ ban đầu
            self.initial_train_test_ratio = current_train_test_ratio
            self.initial_train_valid_ratio = current_train_valid_ratio

        # Bắt đầu tiến trình huấn luyện
        output_dir = "../Model/YOLOv10/"
        data_yaml_path = os.path.join(train_path, "data.yaml")
        os.makedirs(output_dir, exist_ok=True)

        self.stop_event = multiprocessing.Event()

        if selected_model == "YOLOv10":
            self.train_process = multiprocessing.Process(
                target=train_yolov10,
                args=(self.stop_event, data_yaml_path, output_dir, 16, 120) # train với batch size và số epochs
            )
        else:
            tk.messagebox.showerror("Lỗi", f"Mô hình '{selected_model}' chưa được hỗ trợ.")
            return

        # Chuyển trạng thái sang đang huấn luyện
        self.is_training = True
        self.control_button.config(text="Dừng", bootstyle="danger-outline")
        self.train_process.start()

        # Kiểm tra trạng thái tiến trình
        self.check_training_status()

    def stop_training(self):
        """Dừng tiến trình huấn luyện."""
        if self.train_process and self.train_process.is_alive():
            self.stop_event.set()  # Gửi tín hiệu dừng
            self.train_process.terminate()  # Dừng tiến trình
            self.train_process.join()  # Chờ tiến trình kết thúc
            self.is_training = False  # Đặt trạng thái không huấn luyện
            self.control_button.config(text="Bắt đầu", bootstyle="success-outline")
            tk.messagebox.showinfo("Thông báo", "Quá trình huấn luyện đã dừng.")
        else:
            tk.messagebox.showwarning("Cảnh báo", "Không có quá trình huấn luyện nào đang chạy!")

    def check_training_status(self):
        """Kiểm tra trạng thái tiến trình và cập nhật nút điều khiển."""
        if self.train_process and not self.train_process.is_alive():
            # Nếu tiến trình đã kết thúc
            self.is_training = False
            self.control_button.config(text="Bắt đầu", bootstyle="success-outline")
            tk.messagebox.showinfo("Thông báo", "Quá trình huấn luyện đã hoàn tất.")
        else:
            # Tiếp tục kiểm tra sau 500ms
            self.root.after(500, self.check_training_status)

    ###################################
    ###################################
    ####################################
    ##### Phần hiển thị các thông số đánh giá của mô hình
    ####################################
    ####################################

    def show_model_results(self):
        """Hiển thị giao diện chọn thư mục và kết quả mô hình."""
        self.clear_right_frame()

        # Tạo khung chọn thư mục
        self.selection_frame = ttk.Frame(self.right_frame,)
        self.selection_frame.pack(fill=tk.X, padx=20, pady=10)

        # Tiêu đề
        ttk.Label(self.selection_frame, text="Kết quả mô hình", font=("Arial", 18, "bold"), bootstyle="danger").pack(pady=20)
        
        # ComboBox để chọn mô hình
        ttk.Label(self.selection_frame, text="Chọn mô hình:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.model_dir_var = StringVar()
        self.model_combo = ttk.Combobox(self.selection_frame, textvariable=self.model_dir_var, state="readonly", width=100)
        self.model_combo.pack(anchor="w", padx=20, pady=5)
        # Nút xem kết quả
        ttk.Button(self.selection_frame, text="Xem kết quả", command=self.calculate_results, bootstyle="primary").pack(anchor="w", padx=20, pady=10)
        # Lấy danh sách các thư mục mô hình trong '../Model'
        self.populate_model_combo()

        # Tạo khung hiển thị kết quả
        self.result_frame = ttk.Frame(self.right_frame,)
        self.result_frame.pack(fill=tk.X, padx=20, pady=10)

    def clear_results(self):
        """Xóa nội dung hiển thị kết quả trong result_frame."""
        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def populate_model_combo(self):
        """Điền danh sách các mô hình vào ComboBox từ thư mục '../Model'."""
        model_base_path = "../Model"
        if not os.path.exists(model_base_path):
            tk.messagebox.showerror("Lỗi", f"Thư mục '{model_base_path}' không tồn tại.")
            return

        model_dirs = ["Chọn mô hình"]
        self.model_paths = {}  # Lưu đường dẫn tương ứng với từng mục trong ComboBox

        # Duyệt qua từng thư mục trong '../Model'
        for model_name in os.listdir(model_base_path):
            model_path = os.path.join(model_base_path, model_name)
            if os.path.isdir(model_path):  # Kiểm tra thư mục
                # Thêm thư mục gốc (ví dụ: YOLOv10)
                model_dirs.append(model_name)
                self.model_paths[model_name] = os.path.join(model_path, "runs", "detect", "train")

                # Duyệt thêm các thư mục con (nếu có)
                sub_dirs = os.listdir(model_path)
                for sub_dir in sub_dirs:
                    sub_dir_path = os.path.join(model_path, sub_dir)
                    if os.path.isdir(sub_dir_path):  # Kiểm tra thư mục con
                        # Kiểm tra xem thư mục con có đủ file cần thiết không
                        results_file = os.path.join(sub_dir_path, "results.csv")
                        args_file = os.path.join(sub_dir_path, "args.yaml")
                        weights_path = os.path.join(sub_dir_path, "weights", "best.pt")
                        if os.path.exists(results_file) and os.path.exists(args_file) and os.path.exists(weights_path):
                            combo_label = f"{model_name} - {sub_dir}"  # Ví dụ: YOLOv10 - train
                            model_dirs.append(combo_label)
                            self.model_paths[combo_label] = sub_dir_path

        # Gán danh sách thư mục vào ComboBox
        self.model_combo["values"] = model_dirs
        if model_dirs:
            self.model_combo.current(0)  # Chọn giá trị đầu tiên mặc định

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
        """Tính toán kết quả từ mô hình đã chọn, đảm bảo thông tin hiển thị ngay cả khi thiếu cột 'time'."""
        selected_model = self.model_dir_var.get()
         # Kiểm tra nếu người dùng chưa chọn mô hình
        if not selected_model or selected_model == "Chọn mô hình":
            tk.messagebox.showerror("Lỗi", "Hãy chọn một mô hình hợp lệ từ danh sách.")
            return

        model_dir = self.model_paths.get(selected_model, None)
        if not model_dir or not os.path.exists(model_dir):
            tk.messagebox.showerror("Lỗi", f"Không tìm thấy thư mục '{model_dir}'.")
            return

        # Tìm file cần thiết
        results_file = os.path.join(model_dir, 'results.csv')
        args_file = os.path.join(model_dir, 'args.yaml')

        if not os.path.exists(results_file) or not os.path.exists(args_file):
            self.clear_results()
            ttk.Label(self.result_frame, text="Không tìm thấy file results.csv hoặc args.yaml.", bootstyle="danger").pack(pady=10)
            return

        # Đọc file results.csv
        try:
            data = pd.read_csv(results_file)
            # Chuẩn hóa tên cột
            data.columns = data.columns.str.strip()
        except pd.errors.EmptyDataError:
            self.clear_results()
            ttk.Label(self.result_frame, text="File results.csv trống hoặc không hợp lệ.", bootstyle="danger").pack(pady=10)
            return
        except Exception as e:
            self.clear_results()
            ttk.Label(self.result_frame, text=f"Lỗi khi đọc file results.csv: {e}", bootstyle="danger").pack(pady=10)
            return

        # Kiểm tra các cột cần thiết
        required_columns = [
            'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.clear_results()
            ttk.Label(
                self.result_frame,
                text=f"Thiếu các cột dữ liệu sau trong results.csv: {', '.join(missing_columns)}",
                bootstyle="danger"
            ).pack(pady=10)
            return

        # Xử lý thiếu cột 'time'
        time_available = 'time' in data.columns

        # Tính toán train_loss, val_loss, object_accuracy
        try:
            data['train_loss'] = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
            data['val_loss'] = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
            data['object_accuracy'] = 2 * (data['metrics/precision(B)'] * data['metrics/recall(B)']) / (
                data['metrics/precision(B)'] + data['metrics/recall(B)']
            )
            data['object_accuracy'] = data['object_accuracy'].fillna(0)
            data['fitness'] = (0.1 * data['metrics/mAP50(B)'] + 0.9 * data['metrics/mAP50-95(B)'])
        except Exception as e:
            self.clear_results()
            ttk.Label(self.result_frame, text=f"Lỗi khi tính toán các chỉ số: {e}", bootstyle="danger").pack(pady=10)
            return

        # Lấy thông tin từ epoch tốt nhất
        try:
            best_epoch = data.loc[data['fitness'].idxmax()]
            if time_available and pd.notna(best_epoch['time']) and best_epoch['time'] > 0:
                best_time_seconds = best_epoch['time']
                hours, remainder = divmod(best_time_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
            else:
                formatted_time = "Không xác định"
        except Exception as e:
            self.clear_results()
            ttk.Label(self.result_frame, text=f"Lỗi khi lấy thông tin epoch tốt nhất: {e}", bootstyle="danger").pack(pady=10)
            return

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
            f"Đường dẫn mô hình: {model_dir}\n"
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

    ##################################################
    ##################################################
    ##################################################
    ###### Test mô hình trên các tập dữ liệu test
    #############################################
    #############################################

    def show_test_frame(self):
        """Hiển thị giao diện để test mô hình."""
        self.clear_right_frame()

        ttk.Label(self.right_frame, text="Test Mô Hình", font=("Arial", 18, "bold"), bootstyle="danger").pack(pady=20)

        # Phần chọn mô hình
        ttk.Label(self.right_frame, text="Chọn Mô Hình:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.model_dir_var = StringVar()
        self.model_combo = ttk.Combobox(self.right_frame, textvariable=self.model_dir_var, state="readonly", width=100)
        self.model_combo.pack(anchor="w", padx=20, pady=5)
        self.populate_test_model_combo()  # Gọi hàm populate mới

         # Phần chọn tập dữ liệu
        ttk.Label(self.right_frame, text="Chọn Tập Dữ Liệu:", bootstyle="info").pack(anchor="w", padx=20, pady=5)
        self.yaml_file_path = StringVar()
        ttk.Entry(self.right_frame, textvariable=self.yaml_file_path, width=100, state="readonly").pack(anchor="w", padx=20, pady=5)
        ttk.Button(self.right_frame, text="Chọn", command=self.select_dataset_path, bootstyle="primary").pack(anchor="w", padx=20, pady=5)

        # Nút bắt đầu test
        ttk.Button(
            self.right_frame,
            text="Bắt đầu Test",
            command=self.start_testing,
            bootstyle="success-outline",
            width=20
        ).pack(anchor="w", padx=20, pady=20)

        # Kết quả hiển thị
        self.result_frame = ttk.Frame(self.right_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def populate_test_model_combo(self):
        """Điền danh sách các mô hình vào ComboBox từ thư mục '../Model'."""
        model_base_path = "../Model"
        if not os.path.exists(model_base_path):
            tk.messagebox.showerror("Lỗi", f"Thư mục '{model_base_path}' không tồn tại.")
            return

        model_dirs = ["Chọn mô hình"]
        self.model_paths = {}  # Lưu đường dẫn tương ứng với từng mục trong ComboBox

        # Duyệt qua từng thư mục trong '../Model'
        for model_name in os.listdir(model_base_path):
            model_path = os.path.join(model_base_path, model_name)
            if os.path.isdir(model_path):
                if model_name == "YOLOv10":
                    # Model chính
                    main_model_path = os.path.join(model_path, "runs", "detect", "train", "weights", "best.pt")
                    if os.path.exists(main_model_path):
                        model_dirs.append(model_name)
                        self.model_paths[model_name] = main_model_path

                    # Các thư mục train, train1, train2, ...
                    for sub_dir in os.listdir(model_path):
                        sub_dir_path = os.path.join(model_path, sub_dir)
                        weights_path = os.path.join(sub_dir_path, "weights", "best.pt")
                        if os.path.isdir(sub_dir_path) and os.path.exists(weights_path):
                            combo_label = f"{model_name} - {sub_dir}"
                            model_dirs.append(combo_label)
                            self.model_paths[combo_label] = weights_path

        # Gán danh sách mô hình vào ComboBox
        self.model_combo["values"] = model_dirs
        if model_dirs:
            self.model_combo.current(0)  # Chọn giá trị đầu tiên mặc định

        # Liên kết sự kiện chọn mô hình
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

    def on_model_selected(self, event):
        """Xử lý sự kiện khi người dùng chọn mô hình từ ComboBox."""
        selected_model = self.model_combo.get()
        if selected_model == "Chọn mô hình":
            tk.messagebox.showinfo("Thông báo", "Vui lòng chọn một mô hình hợp lệ!")
            self.custom_model = None
            return

        if selected_model in self.model_paths:
            model_path = self.model_paths[selected_model]
            try:
                self.custom_model = YOLO(model_path)
                tk.messagebox.showinfo("Thông báo", f"Mô hình '{selected_model}' đã được tải thành công!")
            except Exception as e:
                tk.messagebox.showerror("Lỗi", f"Lỗi khi tải mô hình: {e}")
                self.custom_model = None

    def create_copy_data_yaml(self, original_yaml_path, test_images_path):
        """Tạo file copy_data.yaml thay thế val path bằng test path."""
        copy_yaml_path = os.path.join(os.path.dirname(original_yaml_path), "copy_data.yaml")
        try:
            with open(original_yaml_path, "r") as file:
                yaml_data = yaml.safe_load(file)
            yaml_data["val"] = test_images_path  # Thay đường dẫn val bằng test/images
            with open(copy_yaml_path, "w") as file:
                yaml.dump(yaml_data, file)
            self.copy_yaml_path = copy_yaml_path
        except Exception as e:
            tk.messagebox.showerror("Lỗi", f"Không thể tạo file 'copy_data.yaml': {e}")
            self.copy_yaml_path = None

    def select_dataset_path(self):
        """Xử lý khi người dùng chọn tập dữ liệu."""
        dataset_path = filedialog.askdirectory()  # Chọn thư mục dataset
        if not dataset_path:
            return

        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        test_images_path = os.path.join(dataset_path, "test/images")

        if not os.path.exists(data_yaml_path):
            tk.messagebox.showerror("Lỗi", f"Không tìm thấy file 'data.yaml' trong: {dataset_path}")
            return
        if not os.path.exists(test_images_path):
            tk.messagebox.showerror("Lỗi", f"Không tìm thấy thư mục 'test/images' trong: {dataset_path}")
            return
        # Xóa file copy_data.yaml cũ nếu đã có
        if hasattr(self, "copy_yaml_path") and self.copy_yaml_path and os.path.exists(self.copy_yaml_path):
            os.remove(self.copy_yaml_path)
            print(f"Đã xóa file tạm cũ: {self.copy_yaml_path}")

        # Lưu đường dẫn gốc và tạo file copy_data.yaml
        self.yaml_file_path.set(data_yaml_path)
        self.create_copy_data_yaml(data_yaml_path, test_images_path) 

    def get_test_paths(self, yaml_file_path):
        """Lấy đường dẫn tới tập ảnh và nhãn test."""
        base_path = os.path.dirname(yaml_file_path)
        images_path = os.path.join(base_path, "test/images")
        labels_path = os.path.join(base_path, "test/labels")

        print("IMAGE PATH:", images_path)
        print("LABELS PATH:", labels_path)

        return {'images': images_path, 'labels': labels_path}

    def read_ground_truth_boxes(self, yaml_file_path):
        """Đọc các bounding box từ nhãn ground truth."""
        test_paths = self.get_test_paths(yaml_file_path)
        labels_path = test_paths['labels']
        images_path = test_paths['images']

        ground_truth_boxes = {}
        for label_file in os.listdir(labels_path):
            if not label_file.endswith('.txt'):
                continue

            image_name = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_path, image_name)

            if not os.path.exists(image_path):
                print(f"Image {image_name} not found, skipping.")
                continue

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            boxes = []
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    _, x_center, y_center, width, height = map(float, parts)

                    # Chuyển đổi từ định dạng YOLO sang [x1, y1, x2, y2]
                    x1 = (x_center - width / 2) * image_width
                    y1 = (y_center - height / 2) * image_height
                    x2 = (x_center + width / 2) * image_width
                    y2 = (y_center + height / 2) * image_height

                    boxes.append([x1, y1, x2, y2])

            ground_truth_boxes[image_name] = boxes

        return ground_truth_boxes
    
    def get_predictions_from_test_set(self, yaml_file_path):
        """Lấy bounding box dự đoán từ mô hình YOLO."""
        test_paths = self.get_test_paths(yaml_file_path)
        images_path = test_paths['images']

        predictions = {}
        results = self.custom_model.predict(source=images_path, save=False)

        for result in results:
            image_name = os.path.basename(result.path)
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            predictions[image_name] = {'boxes': boxes, 'confidences': confidences}

        return predictions
    
    def calculate_miou(self, predictions, ground_truth_boxes):
        """Tính toán mIoU giữa các dự đoán và ground truth boxes."""
        ious = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for image_name, gt_boxes in ground_truth_boxes.items():
            pred_boxes = predictions.get(image_name, {}).get('boxes', [])
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue  # Bỏ qua nếu không có bounding boxes hợp lệ
            # Chuyển đổi sang tensor và đảm bảo định dạng (N, 4)
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4).to(device)
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4).to(device)
            if pred_boxes.shape[1] != 4 or gt_boxes.shape[1] != 4:
                continue  # Bỏ qua nếu định dạng không đúng (N, 4)
            # Tính toán IoU
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            # Lấy giá trị IoU lớn nhất cho mỗi box dự đoán
            max_ious = iou_matrix.max(dim=1).values.cpu().numpy()
            ious.extend(max_ious)

        # Tính giá trị trung bình mIoU
        return np.mean(ious) if ious else 0

    def calculate_plate_detection_accuracy(self, predictions, ground_truth_boxes, iou_threshold=0.5):
        """
        Tính tỷ lệ nhận diện biển số dựa trên các khung dự đoán và nhãn ground truth.
        """
        try:
            correct_detections = 0
            total_plates = sum(len(boxes) for boxes in ground_truth_boxes.values())

            for image_name, gt_boxes in ground_truth_boxes.items():
                pred_boxes = predictions.get(image_name, {}).get('boxes', [])
                
                # Chuyển sang tensor
                pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

                # Bỏ qua ảnh nếu không có dự đoán hoặc ground truth
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                # Tính IoU
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                max_ious = iou_matrix.max(dim=0).values  # Lấy giá trị IoU lớn nhất cho mỗi nhãn thực tế

                # Tính số lượng dự đoán đúng
                correct_detections += (max_ious >= iou_threshold).sum().item()

            # Tính tỷ lệ nhận diện
            accuracy = (correct_detections / total_plates) * 100 if total_plates > 0 else 0
            return accuracy

        except Exception as e:
            print(f"Lỗi khi tính tỷ lệ nhận diện biển số: {e}")
            return 0

    def calculate_average_confidence(self, predictions):
        """
        Tính độ tin cậy trung bình từ các bounding box dự đoán.
        """
        confidences = []
        for data in predictions.values():
            confidences.extend(data.get('confidences', []))
        return np.mean(confidences) if confidences else 0

    def start_testing(self):
        """Bắt đầu kiểm tra mô hình."""
        if self.model_combo.get() == "Chọn mô hình trước tiên":
            tk.messagebox.showerror("Lỗi", "Vui lòng chọn một mô hình trước khi bắt đầu kiểm tra!")
            return

        if not hasattr(self, 'copy_yaml_path') or not self.copy_yaml_path:
            tk.messagebox.showerror("Lỗi", "Hãy chọn tập dữ liệu hợp lệ với file 'copy_data.yaml' đã được tạo.")
            return

        yaml_file_path = self.copy_yaml_path

        if not yaml_file_path or not os.path.exists(yaml_file_path):
            tk.messagebox.showerror("Lỗi", f"Không tìm thấy tệp 'copy_data.yaml' tại: {yaml_file_path}")
            return

        # Vô hiệu hóa nút và hiển thị trạng thái
        self.disable_buttons()
        self.status_label = ttk.Label(self.result_frame, text="Đang test mô hình...", font=("Arial", 14), bootstyle="info")
        self.status_label.pack(anchor="w", pady=10)

        # Chạy quá trình kiểm tra trong luồng riêng
        test_thread = threading.Thread(target=self.run_testing, args=(yaml_file_path,))
        test_thread.start()

    def run_testing(self, yaml_file_path):
        """
        Thực hiện kiểm tra mô hình và cập nhật GUI khi hoàn thành.
        """
        try:
            # Kiểm tra tệp YAML
            if not os.path.exists(yaml_file_path):
                self.status_label.config(text="Lỗi: Không tìm thấy tệp 'data.yaml'")
                return

            # Kiểm tra mô hình
            if not hasattr(self, 'custom_model') or self.custom_model is None:
                self.status_label.config(text="Lỗi: Mô hình chưa được chọn.")
                return

            # Tính mAP bằng model.val()
            results = self.custom_model.val(data=yaml_file_path, save_json=True)
            mAP_50 = results.box.map50
            mAP_50_95 = results.box.map

            # Đọc ground truth và dự đoán
            ground_truth_boxes = self.read_ground_truth_boxes(yaml_file_path)
            predictions = self.get_predictions_from_test_set(yaml_file_path)

            # Tính các thông số bổ sung
            mIoU = self.calculate_miou(predictions, ground_truth_boxes)
            plate_detection_accuracy = self.calculate_plate_detection_accuracy(predictions, ground_truth_boxes)
            average_confidence = self.calculate_average_confidence(predictions)

            # Tạo kết quả
            result_text = (
                f"Kết quả kiểm tra mô hình:\n"
                f"--------------------------\n"
                f"mAP@0.5: {mAP_50:.4f}\n"
                f"mAP@0.5:0.95: {mAP_50_95:.4f}\n"
                f"mIoU: {mIoU:.4f}\n"
                f"Độ tin cậy trung bình: {average_confidence:.2f}\n"
                f"Tỷ lệ nhận diện biển số: {plate_detection_accuracy:.2f}%\n"
            )

            # Cập nhật GUI
            self.status_label.config(text=result_text, bootstyle="success")

        except Exception as e:
            # Hiển thị lỗi trên GUI
            self.status_label.config(text=f"Lỗi khi kiểm tra mô hình: {e}", bootstyle="danger")

        finally:
            # Kích hoạt lại các nút
            self.enable_buttons()

    def disable_buttons(self):
        """Đóng băng tất cả các nút trong giao diện."""
        for widget in self.right_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state="disabled")

    def enable_buttons(self):
        """Kích hoạt lại tất cả các nút trong giao diện."""
        for widget in self.right_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state="normal")


    # def clear_results(self):
    #     """Xóa nội dung hiển thị kết quả trong result_frame."""
    #     for widget in self.result_frame.winfo_children():
    #         widget.destroy()

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

    ####################################################
    ###################################################
    ####### Phần detect video/ảnh hay demo 
    #######################################################
    ######################################################
    ######################################################

    def show_detect_content(self):
        """Hiển thị nội dung hiện tại của Detect video và ảnh."""
        self.clear_right_frame()

        # Khởi tạo OCR khi vào giao diện này
        if not hasattr(self, 'ocr') or self.ocr is None:
            try:
                self.ocr = PaddleOCR(lang='en')  # Khởi tạo OCR
            except Exception as e:
                tk.messagebox.showerror("Lỗi", f"Không thể khởi tạo OCR: {e}")
                return

        # Tạo khung giao diện như bình thường
        self.top_controls = tk.Frame(self.right_frame, bg="#d3d3d3")
        self.top_controls.pack(side=tk.TOP, anchor='w', pady=10)

         # Phần chọn model bằng ComboBox
        ttk.Label(self.top_controls, text="Chọn Mô Hình:", bootstyle="info").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.detect_model_var = StringVar()
        self.detect_model_combo = ttk.Combobox(self.top_controls, textvariable=self.detect_model_var, state="readonly", width=50)
        self.detect_model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.populate_detect_model_combo()  # Gọi hàm để điền danh sách mô hình

        # Liên kết sự kiện chọn mô hình
        self.detect_model_combo.bind("<<ComboboxSelected>>", self.on_detect_model_selected)

        # Phần chọn video
        ttk.Label(self.top_controls, text="Chọn Video hoặc Ảnh:", bootstyle="info").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.video_path_label = ttk.Label(self.top_controls, text="No video selected", bootstyle="secondary-inverse", anchor="w")
        self.video_path_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.select_video_icon = ImageTk.PhotoImage(Image.open("./img/select_video.png").resize((30, 30), Image.LANCZOS))
        self.btn_select = tk.Button(self.top_controls, image=self.select_video_icon, command=self.load_media)
        self.btn_select.grid(row=1, column=2, padx=10, pady=5, sticky="w")

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

    def populate_detect_model_combo(self):
        """Điền danh sách các mô hình vào ComboBox từ thư mục '../Model'."""
        model_base_path = "../Model"
        if not os.path.exists(model_base_path):
            tk.messagebox.showerror("Lỗi", f"Thư mục '{model_base_path}' không tồn tại.")
            return

        model_dirs = ["Chọn mô hình"]
        self.detect_model_paths = {}

        for model_name in os.listdir(model_base_path):
            model_path = os.path.join(model_base_path, model_name)
            if os.path.isdir(model_path):
                # Model chính
                main_model_path = os.path.join(model_path, "runs", "detect", "train", "weights", "best.pt")
                if os.path.exists(main_model_path):
                    model_dirs.append(model_name)
                    self.detect_model_paths[model_name] = main_model_path

                # Các thư mục con
                for sub_dir in os.listdir(model_path):
                    sub_dir_path = os.path.join(model_path, sub_dir)
                    weights_path = os.path.join(sub_dir_path, "weights", "best.pt")
                    if os.path.isdir(sub_dir_path) and os.path.exists(weights_path):
                        combo_label = f"{model_name} - {sub_dir}"
                        model_dirs.append(combo_label)
                        self.detect_model_paths[combo_label] = weights_path

        # Gán danh sách vào ComboBox
        self.detect_model_combo["values"] = model_dirs
        self.detect_model_combo.current(0)  # Mặc định chọn "Chọn mô hình"

    def on_detect_model_selected(self, event):
        """Xử lý sự kiện khi người dùng chọn mô hình từ ComboBox."""
        selected_model = self.detect_model_combo.get()
        if selected_model == "Chọn mô hình":
            tk.messagebox.showinfo("Thông báo", "Vui lòng chọn một mô hình hợp lệ!")
            self.detect_model = None
            return

        if selected_model in self.detect_model_paths:
            model_path = self.detect_model_paths[selected_model]
            try:
                self.detect_model = YOLO(model_path)
                tk.messagebox.showinfo("Thông báo", f"Mô hình '{selected_model}' đã được tải thành công!")
            except Exception as e:
                tk.messagebox.showerror("Lỗi", f"Lỗi khi tải mô hình: {e}")
                self.detect_model = None

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
        if not hasattr(self, 'canvas_video') or self.canvas_video is None:
            tk.messagebox.showerror("Lỗi", "Canvas video chưa được khởi tạo. Vui lòng chuyển sang giao diện 'Detect video và ảnh'.")
            return

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

        # Lưu ảnh vào thuộc tính để giữ tham chiếu và tránh bị thu hồi bộ nhớ
        self.current_image = img_tk

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
            class_name = self.detect_model.names[class_id]
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
        results = self.detect_model(image)
        detected_objects = results[0].boxes
        # Xóa các widget cũ trong khung cuộn trước khi hiển thị kết quả mới
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        # Hiển thị các đối tượng được phát hiện trong khung cuộn
        for i, box in enumerate(detected_objects[:3]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.detect_model.names[class_id]

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
        # Sử dụng mô hình được chọn nếu có, nếu không thì thông báo lỗi
        if not hasattr(self, 'detect_model') or self.detect_model is None:
            tk.messagebox.showerror("Lỗi", "Mô hình chưa được khởi tạo hoặc chọn. Vui lòng chọn mô hình trước.")
            return {}

        try:
            # Phát hiện đối tượng với mô hình được chọn
            results = self.detect_model(frame)  # YOLO detection
            best_detections = self.get_best_detections(results[0].boxes, frame)

            # Xử lý OCR cho từng vùng được cắt
            for key, data in best_detections.items():
                cropped_img = data['image']
                ocr_results = self.ocr.ocr(cropped_img, cls=True)
                ocr_text = self.extract_ocr_text(ocr_results)  # Lấy văn bản OCR

                # Thêm văn bản OCR vào phát hiện
                best_detections[key]['ocr_text'] = ocr_text

            return best_detections
        except Exception as e:
            tk.messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi phát hiện đối tượng: {e}")
            return {}
    
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
        results = self.detect_model(frame)
        detected_objects = results[0].boxes

        # Xóa các widget cũ trong khung cuộn trước khi hiển thị kết quả mới
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Hiển thị các đối tượng được phát hiện trong khung cuộn
        for i, box in enumerate(detected_objects[:3]):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = self.detect_model.names[class_id]

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
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.mainloop()