# Hướng dẫn cài đặt và chạy chương trình nhận diện biển số xe qua video:
# Yêu cầu máy phải có GPU tương thích
# Cài đặt NVIDIA Toolkit phiên bản 11.8 (Yêu cầu có GPU tương thích)
# Python 3.11.9
# Cài đặt thư viện Pytorch hỗ trợ phiên bản cuda 11.8

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt thư viện từ file requirements.txt
pip install -r requirements.txt

# Vào thư mục PythonApp.py trong thư mục CarDetection/App, mở Command Prompt tại thư mục và chạy lệnh 
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


	