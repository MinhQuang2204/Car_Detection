# Hướng dẫn cài đặt và chạy chương trình nhận diện biển số xe qua video:
# Yêu cầu máy phải có GPU tương thích
# Cài đặt NVIDIA Toolkit phiên bản 11.8 (Yêu cầu có GPU tương thích)

# Python 3.11.9

# Tại thư mục Car_Detection\App tạo môi trường ảo, mở command prompt
# Nếu thư mục đặt ở ổ đĩa khác C, gõ (tên ổ đĩa): ; ví dụ D:
# dùng lệnh cd để dẫn đến thư mục Car_Detection
# gõ lệnh để tạo môi trường ảo

python -m venv .venv

# thư mục .venv chứa môi trường ảo được tạo ra tại thư mục Car_Detection
# gõ lệnh để kích hoạt môi trường ảo

.venv\Scripts\activate.bat

# Cài đặt thư viện từ file requirements.txt
pip install -r requirements.txt

# chạy lệnh sau để kiểm tra xem máy có thể sử dụng thư viện torch trên gpu không, nếu máy có gpu nvidia
nvidia-smi

# Nếu chạy được lệnh trên và hiển thị kết quả Driver Version >= 450.80.02 thì có thể cài đặt thư viện Pytorch hỗ trợ GPU với cuda 11.8, chạy lệnh này để cài đặt

pip uninstall torch torchvision torchaudio -y

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Trong trường hợp cài đặt được (Pytorch + cuda) nhưng chạy bị lỗi có thể gỡ các thư viện pytorch ra và cài đặt lại các thư viện có trong gói Ultralytics để chuyển sang sử dụng cpu

pip uninstall torch torchvision torchaudio -y

pip install ultralytics

# cd vào trong thư mục App trong CarDetection/App, chạy lệnh
python PythonApp.py

# Thực hiện các thao tác trên giao diện

# Lưu ý:
# Thư mục file dataset cần tuân thủ cấu trúc theo định dạng yolov8
# Cấu trúc thư mục như sau
# Dataset
# 	|__ train
#	|	|__ images
#	|	|__ labels
#	|__ valid
#	|	|__ images
#	|	|__ labels
#	|__ test
#	|	|__ images
#	|	|__ labels
#	|__ data.yaml
# Hoặc ít nhất là cấu trúc sau:
#
# Dataset 
#	|__ images
#	|__ labels
#	|__ data.yaml


	