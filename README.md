# ProLabeler
#  ProLabeler v1.1
**Smart, automatic video-to-dataset tool for YOLO models (ONNX format)**

ProLabeler turns any video (local or YouTube) into a clean, ready-to-train YOLO dataset.  
Supports multi-model ensemble detection, GPU acceleration, and automatic label formatting.

## To Do
-  Add multi-class support
-  Video Preview with OpenCV VideoWriter
-  Drag and Drop support

---

##  Features
-  Works with **YOLOv8 ONNX models** (shape `1×5×8400`)
-  Input from **MP4** or **YouTube links**
-  GPU acceleration (CUDA) with automatic CPU fallback
-  Ensemble detection with box merging
-  Automatic label scaling and aspect-safe resizing
-  Clean GUI built with **Tkinter + Sun Valley ttk**
-  Save & load settings (`config.json`)
-  Auto Skips simmilar frames

---

##  Quick Start
###  Option 1: Use the compiled `.exe`
Just download the latest release — no installation needed.


###  Option 2: Run from source
####  Prerequisites
- **Python 3.10+**
- **pip** and **venv** installed
- *(Optional)* NVIDIA GPU + CUDA 12 Toolkit for GPU acceleration
- FFMPEG is required, but it is bundled with this program

  
Create and activate a virtual environment
  ```
  python -m venv pro
  ```
  ### Windows
  ```
  pro\Scripts\activate
  ```
  ### macOS / Linux
  ```
  source pro/bin/activate
  ```
  

### CPU Version
```
   pip install .[cpu]
```

### GPU Version (requires CUDA)
```
   pip install .[gpu]
```
GPU mode uses onnxruntime-gpu for better performance.
But you need CUDA 12.x installed
### Launch the GUI
```
python App.py
```
### Or use the CLI version
Run detections directly via terminal:
```
python Main.py --video path/to/video.mp4 \
               --models path/to/model1.onnx path/to/model2.onnx \
               --out ./output \
               --conf 0.5 \
               --iou 0.45 \
               --frame-step 1 \
               --merge-iou 0.6
```
Example (YouTube video):
```
python Main.py --youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
               --yt-res 1080p \
               --models ./models/my_model.onnx \
               --out ./output \
               --conf 0.5
```
Add --cpu to force CPU mode.
