"""Step 0: 설치가 잘 됐는지 확인. python3 code/mono_depth/step0_check.py"""
import cv2
import numpy
import open3d
import torch
import transformers

print("cv2", cv2.__version__, "| numpy", numpy.__version__, "| open3d", open3d.__version__)
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda:", torch.cuda.is_available())
cap = cv2.VideoCapture(0)
print("webcam:", cap.isOpened())
cap.release()
