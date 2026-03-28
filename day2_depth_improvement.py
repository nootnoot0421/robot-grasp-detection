
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline
from PIL import Image

model = YOLO("yolov8n.pt")
depth_pipe = hf_pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

img = Image.open("test.jpg")
results = model("test.jpg")
depth_map = np.array(depth_pipe(img)["depth"])

box = results[0].boxes[0]
x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
cx, cy = (x1+x2)//2, (y1+y2)//2

# 原来的做法
depth_old = depth_map[cy, cx]

# 改进：取框内最近10%过滤背景
roi = depth_map[y1:y2, x1:x2]
depth_new = np.percentile(roi, 10)

print(f"原来（中心点）: {depth_old}")
print(f"改进（过滤背景）: {depth_new:.1f}")
print(f"差值: {depth_new - depth_old:.1f}")
