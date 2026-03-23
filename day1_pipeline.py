
import cv2, torch, numpy as np
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline
from PIL import Image

# YOLO检测
model = YOLO("yolov8n.pt")
results = model("test.jpg")
result_img = results[0].plot()
cv2.imwrite("output.jpg", result_img)

# 深度估计
depth_pipe = hf_pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
img = Image.open("test.jpg")
depth_map = np.array(depth_pipe(img)["depth"])
depth_colored = cv2.applyColorMap(
    (depth_map / depth_map.max() * 255).astype(np.uint8),
    cv2.COLORMAP_MAGMA
)
cv2.imwrite("depth_output.jpg", depth_colored)
print("完成")
