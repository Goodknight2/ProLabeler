# ProLabeler
#  ProLabeler v1.0
**Smart, automatic video-to-dataset tool for YOLO models (ONNX format)**

ProLabeler turns any video (local or YouTube) into a clean, ready-to-train YOLO dataset.  
Supports multi-model ensemble detection, GPU acceleration, and automatic label formatting.

## To Do
Add multi-class support
Video Preview with OpenCV VideoWriter
Drag and Drop support
Add Tooltips in the GUI

---

##  Features
-  Works with **YOLOv8 ONNX models** (shape `1×5×8400`)
-  Input from **MP4** or **YouTube links**
-  GPU acceleration (CUDA) with automatic CPU fallback
-  Ensemble detection with box merging
-  Automatic label scaling and aspect-safe resizing
-  Clean GUI built with **Tkinter + Sun Valley ttk**
-  Save & load settings (`config.json`)

---

##  Quick Start
###  Option 1: Use the compiled `.exe`
Just run **`ProLabeler.exe`** — no installation needed.

###  Option 2: Run from source
```bash
# Step 1: Create and activate a venv
python -m venv pro
pro\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step :3 Run the GUI
python App.py
