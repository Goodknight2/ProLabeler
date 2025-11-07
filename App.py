import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import sv_ttk
import subprocess
import threading

CONFIG_FILE = "config.json"

class ToolTip:
    """Creates a tooltip for a given widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                        background="#2b2b2b", foreground="#ffffff",
                        relief='solid', borderwidth=1,
                        font=("Segoe UI", 9), padx=8, pady=6)
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("ProLabeler GUI")
        self.geometry("950x650")
        self.minsize(900, 630)

        sv_ttk.set_theme("dark")  # Enable Sun Valley Dark Theme

        # Initialize model path mapping
        self.model_paths = {}

        # --- Layout setup ---
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.left_frame = ttk.Frame(self, padding=10)
        self.right_frame = ttk.Frame(self, padding=10)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # --- Left side ---
        video_label = ttk.Label(self.left_frame, text="Video Input")
        video_label.grid(row=0, column=0, sticky="w")
        ToolTip(video_label, "Select a local video file to process")
        
        self.video_path = tk.StringVar()
        video_frame = ttk.Frame(self.left_frame)
        video_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_path)
        self.video_entry.pack(side="left", fill="x", expand=True)
        ToolTip(self.video_entry, "Path to your video file (MP4, AVI, MOV)")
        
        browse_btn = ttk.Button(video_frame, text="Browse", command=self.browse_video)
        browse_btn.pack(side="right", padx=5)
        ToolTip(browse_btn, "Browse for a video file on your computer")

        yt_label = ttk.Label(self.left_frame, text="YouTube Link (optional)")
        yt_label.grid(row=2, column=0, sticky="w")
        ToolTip(yt_label, "Download and process a video from YouTube instead")
        
        self.youtube_link = tk.StringVar()
        yt_entry = ttk.Entry(self.left_frame, textvariable=self.youtube_link)
        yt_entry.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ToolTip(yt_entry, "Paste a YouTube URL here (overrides video input)")

        yt_res_label = ttk.Label(self.left_frame, text="YouTube Resolution")
        yt_res_label.grid(row=4, column=0, sticky="w")
        ToolTip(yt_res_label, "Maximum resolution to download from YouTube")
        
        self.yt_res = ttk.Combobox(self.left_frame, values=["360p", "480p", "720p", "1080p", "1440p", "2160p"])
        self.yt_res.current(3)
        self.yt_res.grid(row=5, column=0, sticky="ew", pady=(0, 15))
        ToolTip(self.yt_res, "Higher resolution = better quality but larger download")

        out_res_label = ttk.Label(self.left_frame, text="Output Image Resolution")
        out_res_label.grid(row=6, column=0, sticky="w")
        ToolTip(out_res_label, "Resize saved images (Off = keep original size)")
        
        self.out_res = ttk.Combobox(
            self.left_frame,
            values=["Off", "640x640", "320x320", "1280x1280"],
        )
        self.out_res.current(0)
        self.out_res.grid(row=7, column=0, sticky="ew", pady=(0, 15))
        ToolTip(self.out_res, "640x640 is standard for YOLOv8 training")

        models_label = ttk.Label(self.left_frame, text="ONNX Models (multiple allowed)")
        models_label.grid(row=9, column=0, sticky="w")
        ToolTip(models_label, "Add one or more YOLO ONNX models for detection")
        
        model_frame = ttk.Frame(self.left_frame)
        model_frame.grid(row=10, column=0, sticky="ew", pady=(0, 10))
        self.models_listbox = tk.Listbox(model_frame, height=5, selectmode="multiple", bg="#1e1e1e", fg="#f0f0f0")
        self.models_listbox.pack(side="left", fill="both", expand=True)
        
        ttk.Scrollbar(model_frame, command=self.models_listbox.yview).pack(side="right", fill="y")
        
        model_button_frame = ttk.Frame(self.left_frame)
        model_button_frame.grid(row=11, column=0, sticky="ew", pady=(0, 15))
        
        add_model_btn = ttk.Button(model_button_frame, text="Add Model", command=self.add_model)
        add_model_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ToolTip(add_model_btn, "Browse for .onnx model files")
        
        remove_model_btn = ttk.Button(model_button_frame, text="Remove Selected", command=self.remove_model)
        remove_model_btn.pack(side="left", fill="x", expand=True)
        ToolTip(remove_model_btn, "Remove selected models from the list")

        output_label = ttk.Label(self.left_frame, text="Output Folder")
        output_label.grid(row=12, column=0, sticky="w")
        ToolTip(output_label, "Where to save labeled images and annotations")
        
        self.output_path = tk.StringVar()
        output_frame = ttk.Frame(self.left_frame)
        output_frame.grid(row=13, column=0, sticky="ew", pady=(0, 15))
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        self.output_entry.pack(side="left", fill="x", expand=True)
        ToolTip(self.output_entry, "Automatically creates images and labels subfolders")
        
        output_browse_btn = ttk.Button(output_frame, text="Browse", command=self.browse_output)
        output_browse_btn.pack(side="right", padx=5)
        ToolTip(output_browse_btn, "Select output directory")

        # --- Right side ---
        ttk.Label(self.right_frame, text="Settings").grid(row=0, column=0, sticky="w")

        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.frame_step = tk.IntVar(value=1)
        self.merge_iou = tk.DoubleVar(value=0.6)
        self.use_gpu = tk.BooleanVar(value=True)
        self.similarity_threshold = tk.IntVar(value=5)
        self.history_size = tk.IntVar(value=30)

        # Settings with tooltips
        settings_config = [
            ("Confidence Threshold", self.conf_threshold, 0.01, 1.0, 0.01, 
             "Minimum confidence score to keep a detection\nHigher = fewer but more confident detections"),
            ("IoU Threshold", self.iou_threshold, 0.01, 1.0, 0.01,
             "IoU threshold for Non-Maximum Suppression\nHigher = more overlapping boxes allowed"),
            ("Frame Step", self.frame_step, 1, 30, 1,
             "Process every frame\n1 = every frame, 2 = every other frame, etc."),
            ("Box Merge IoU", self.merge_iou, 0.01, 1.0, 0.01,
             "IoU threshold for merging boxes from multiple models\nHigher = only merge very overlapping boxes"),
            ("Duplicate Skip Threshold", self.similarity_threshold, 0, 64, 1,
             "How similar frames can be before skipping\n0 = identical only, 5 = very similar (recommended)\n10-15 = Kinda similar, 20+ = Meh"),
            ("History Size", self.history_size, 5, 100, 1,
             "Number of frames to compare against\nHigher = catches duplicates further apart\nLower = faster processing"),
        ]

        for i, (label, var, a, b, step, tooltip) in enumerate(settings_config, start=1):
            label_widget = ttk.Label(self.right_frame, text=label)
            label_widget.grid(row=i, column=0, sticky="w")
            ToolTip(label_widget, tooltip)

            if isinstance(var, tk.DoubleVar):
                scale_frame = ttk.Frame(self.right_frame)
                scale_frame.grid(row=i, column=1, sticky="ew", padx=(10, 0))

                # The scale itself
                scale = ttk.Scale(scale_frame, variable=var, from_=a, to=b, orient="horizontal", length=150)
                scale.pack(side="left", fill="x", expand=True)
                ToolTip(scale, tooltip)

                # Value label that updates live
                val_label = ttk.Label(scale_frame, text=f"{var.get():.2f}", width=5)
                val_label.pack(side="right", padx=(8, 0))

                # Update label as slider moves
                def update_label(val, lbl=val_label):
                    lbl.config(text=f"{float(val):.2f}")
                scale.config(command=update_label)

            else:
                spinbox = ttk.Spinbox(self.right_frame, from_=a, to=b, textvariable=var, width=5)
                spinbox.grid(row=i, column=1, sticky="ew", padx=(10, 0))
                ToolTip(spinbox, tooltip)

        gpu_check = ttk.Checkbutton(
            self.right_frame,
            text="Use GPU (fallback to CPU if unavailable)",
            variable=self.use_gpu,
        )
        gpu_check.grid(row=7, column=0, columnspan=2, pady=(15, 10))
        ToolTip(gpu_check, "Use CUDA GPU acceleration if available\nAutomatically falls back to CPU if GPU not detected")

        save_config_btn = ttk.Button(self.right_frame, text="Save Config", command=self.save_config)
        save_config_btn.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(5, 5))
        ToolTip(save_config_btn, "Save current settings to config.json\nSettings will load automatically next time")
        
        # Run Detection Button
        run_btn = ttk.Button(self.right_frame, text="Run Detection", command=self.run_detection)
        run_btn.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(5, 10))
        ToolTip(run_btn, "Start processing the video with current settings")

        # Log Output Label + Box
        log_label = ttk.Label(self.right_frame, text="Log Output")
        log_label.grid(row=10, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ToolTip(log_label, "Real-time output from the detection process")
        
        self.log_text = tk.Text(self.right_frame, height=8, wrap="word")
        self.log_text.grid(row=11, column=0, columnspan=2, sticky="nsew", pady=(0, 5))
        self.log_text.configure(bg="#1e1e1e", fg="#f0f0f0", insertbackground="#f0f0f0", selectbackground="#3a3a3a")

        # Status Indicator (moved below log box)
        self.status_label = ttk.Label(
            self.right_frame,
            text="🟢 Ready",
            foreground="#9FEF9F",  # light green
            font=("Segoe UI", 10, "bold")
        )
        self.status_label.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(5, 5))
        ToolTip(self.status_label, "Current processing status")
        
        # Progress bar (hidden until used)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.right_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate"
        )
        self.progress_bar.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.progress_bar.grid_remove()  # start hidden

        # Detection progress bar (hidden until used)
        self.detect_progress_var = tk.DoubleVar(value=0)
        self.detect_progress_bar = ttk.Progressbar(
            self.right_frame,
            variable=self.detect_progress_var,
            maximum=100,
            mode="determinate"
        )
        self.detect_progress_bar.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.detect_progress_bar.grid_remove()  # hidden by default

        # ETA Label for detection
        self.eta_label = ttk.Label(self.right_frame, text="")
        self.eta_label.grid(row=13, column=0, columnspan=2, sticky="ew")

        # Cancel Button
        self.cancel_flag = False
        self.cancel_button = ttk.Button(self.right_frame, text="Cancel Detection", command=self.cancel_detection)
        self.cancel_button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(5, 15))
        self.cancel_button.grid_remove()
        ToolTip(self.cancel_button, "Stop the detection process early")

        # Let the log box expand properly
        self.right_frame.rowconfigure(11, weight=1)

        self.load_config()

    # ---------- File Choosers ----------
    def browse_video(self):
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path.set(file_path)

    def add_model(self):
        files = filedialog.askopenfilenames(title="Select ONNX Models", filetypes=[("ONNX Model", "*.onnx")])
        for f in files:
            # Store full path but display only filename
            filename = os.path.basename(f)
            self.models_listbox.insert("end", filename)
            # Store the full path in a mapping
            if not hasattr(self, 'model_paths'):
                self.model_paths = {}
            self.model_paths[filename] = f

    def remove_model(self):
        selected = self.models_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select one or more models to remove.")
            return
        # Delete in reverse order to avoid index shifting issues
        for index in reversed(selected):
            filename = self.models_listbox.get(index)
            self.models_listbox.delete(index)
            # Remove from mapping
            if filename in self.model_paths:
                del self.model_paths[filename]

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)
            
    def update_status(self, text, color):
        """Update the status indicator text and color"""
        self.status_label.config(text=text, foreground=color)
        self.status_label.update_idletasks()

    def update_progress(self, percent):
        try:
            self.progress_var.set(percent)
            self.progress_bar.update_idletasks()
        except:
            pass

    def cancel_detection(self):
        """Set a flag that tells Main.py to stop detection early."""
        self.cancel_flag = True
        self.update_status("🛑 Canceling...", "#FF7F7F")

    def update_detection_progress(self, percent, eta=None):
        """Update the detection progress bar and ETA label."""
        try:
            if percent is not None:
                self.detect_progress_var.set(percent)
                self.detect_progress_bar.update_idletasks()

            if eta:
                self.eta_label.config(text=f"ETA: {eta:.1f}s remaining")
            else:
                self.eta_label.config(text="")
        except:
            pass

    # ---------- Config Management ----------
    def save_config(self):
        # Get full paths for models
        model_filenames = list(self.models_listbox.get(0, "end"))
        full_model_paths = [self.model_paths.get(fn, fn) for fn in model_filenames]
        
        config = {
            "video_path": self.video_path.get(),
            "youtube_link": self.youtube_link.get(),
            "yt_res": self.yt_res.get(),
            "models": full_model_paths,
            "output_path": self.output_path.get(),
            "conf_threshold": self.conf_threshold.get(),
            "iou_threshold": self.iou_threshold.get(),
            "frame_step": self.frame_step.get(),
            "merge_iou": self.merge_iou.get(),
            "use_gpu": self.use_gpu.get(),
            "similarity_threshold": self.similarity_threshold.get(),
            "history_size": self.history_size.get(),
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        messagebox.showinfo("Saved", "Configuration saved successfully!")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            self.video_path.set(config.get("video_path", ""))
            self.youtube_link.set(config.get("youtube_link", ""))
            self.yt_res.set(config.get("yt_res", "1080p"))
            self.out_res.set(config.get("out_res", "off"))
            self.models_listbox.delete(0, "end")
            self.model_paths = {}
            for full_path in config.get("models", []):
                filename = os.path.basename(full_path)
                self.models_listbox.insert("end", filename)
                self.model_paths[filename] = full_path
            self.output_path.set(config.get("output_path", ""))
            self.conf_threshold.set(config.get("conf_threshold", 0.5))
            self.iou_threshold.set(config.get("iou_threshold", 0.45))
            self.frame_step.set(config.get("frame_step", 1))
            self.merge_iou.set(config.get("merge_iou", 0.6))
            self.use_gpu.set(config.get("use_gpu", True))
            self.similarity_threshold.set(config.get("similarity_threshold", 5))
            self.history_size.set(config.get("history_size", 30))

    class TextRedirector:
        """Redirects print() output to a Tkinter Text widget in real time."""
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, message):
            self.text_widget.insert("end", message)
            self.text_widget.see("end")
            self.text_widget.update_idletasks()

        def flush(self):
            pass  # Needed for compatibility with sys.stdout

    def run_detection(self):
        model_filenames = list(self.models_listbox.get(0, "end"))
        if not model_filenames:
            messagebox.showerror("Error", "Please add at least one ONNX model.")
            return

        models = [self.model_paths.get(fn, fn) for fn in model_filenames]
        self.cancel_flag = False
        self.detect_progress_bar.grid_remove()
        self.progress_bar.grid_remove()
        self.detect_progress_var.set(0)
        self.eta_label.config(text="")
        self.cancel_button.grid_remove()
        self.run_button_state(False)
        self.update_status("🟡 Running...", "#FFD966")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "Starting detection...\n")
        self.log_text.see("end")

        # Prepare arguments as if from CLI
        args = {
            "video": self.video_path.get() or None,
            "youtube": self.youtube_link.get() or None,
            "yt_res": self.yt_res.get(),
            "out_res": self.out_res.get() if self.out_res.get().lower() != "off" else None,
            "models": models,
            "out": self.output_path.get() or "./output",
            "conf": self.conf_threshold.get(),
            "iou": self.iou_threshold.get(),
            "frame_step": self.frame_step.get(),
            "merge_iou": self.merge_iou.get(),
            "cpu": not self.use_gpu.get(),
            "similarity_threshold": self.similarity_threshold.get(),
            "history_size": self.history_size.get(),
        }

        threading.Thread(target=self.run_internal_main, args=(args,), daemon=True).start()
        self.cancel_button.grid()


    def yt_progress_callback(self, text=None, percent=None):
        if text:
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")

        if percent is not None:
            # Show bar if hidden
            self.progress_bar.grid()
            self.update_progress(percent)

        # Hide bar when finished
        if text and "Download complete" in text:
            self.progress_var.set(100)


    def run_internal_main(self, args):
        """Run Main.py directly and stream print() output to the log box."""
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "Starting detection...\n\n")

        try:
            import sys
            import Main

            # Save original stdout/stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr

            # Redirect print output to text box
            sys.stdout = sys.stderr = self.TextRedirector(self.log_text)

            self.update_status("🟡 Running...", "#FFD966")

            # Run the main process directly
            Main.main(
                args=args,
                progress_callback=self.yt_progress_callback,
                detect_progress_callback=self.update_detection_progress,
                cancel_check=lambda: self.cancel_flag
            )

            self.update_status("🟢 Done", "#9FEF9F")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            self.update_status("🔴 Error", "#FF7F7F")

        finally:
            # Restore console output
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.run_button_state(True)
            print("\nProcess finished.\n")



    def run_button_state(self, enabled: bool):
        """Enable or disable the Run button"""
        for child in self.right_frame.winfo_children():
            if isinstance(child, ttk.Button) and "Run Detection" in child.cget("text"):
                child.config(state="normal" if enabled else "disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()