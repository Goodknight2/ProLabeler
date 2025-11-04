import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import sv_ttk
import subprocess
import threading

CONFIG_FILE = "config.json"

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("ProLabeler GUI")
        self.geometry("950x620")
        self.minsize(900, 600)

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
        ttk.Label(self.left_frame, text="Video Input").grid(row=0, column=0, sticky="w")
        self.video_path = tk.StringVar()
        video_frame = ttk.Frame(self.left_frame)
        video_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_path)
        self.video_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(video_frame, text="Browse", command=self.browse_video).pack(side="right", padx=5)

        ttk.Label(self.left_frame, text="YouTube Link (optional)").grid(row=2, column=0, sticky="w")
        self.youtube_link = tk.StringVar()
        ttk.Entry(self.left_frame, textvariable=self.youtube_link).grid(row=3, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(self.left_frame, text="YouTube Resolution").grid(row=4, column=0, sticky="w")
        self.yt_res = ttk.Combobox(self.left_frame, values=["360p", "480p", "720p", "1080p", "1440p", "2160p"])
        self.yt_res.current(3)
        self.yt_res.grid(row=5, column=0, sticky="ew", pady=(0, 15))

        ttk.Label(self.left_frame, text="Output Image Resolution").grid(row=6, column=0, sticky="w")
        self.out_res = ttk.Combobox(
            self.left_frame,
            values=["Off", "640x640", "320x320", "1280x1280"],
        )
        self.out_res.current(0)
        self.out_res.grid(row=7, column=0, sticky="ew", pady=(0, 15))


        ttk.Label(self.left_frame, text="ONNX Models (multiple allowed)").grid(row=9, column=0, sticky="w")
        model_frame = ttk.Frame(self.left_frame)
        model_frame.grid(row=10, column=0, sticky="ew", pady=(0, 10))
        self.models_listbox = tk.Listbox(model_frame, height=5, selectmode="multiple", bg="#1e1e1e", fg="#f0f0f0")
        self.models_listbox.pack(side="left", fill="both", expand=True)
        ttk.Scrollbar(model_frame, command=self.models_listbox.yview).pack(side="right", fill="y")
        
        model_button_frame = ttk.Frame(self.left_frame)
        model_button_frame.grid(row=11, column=0, sticky="ew", pady=(0, 15))
        ttk.Button(model_button_frame, text="Add Model", command=self.add_model).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(model_button_frame, text="Remove Selected", command=self.remove_model).pack(side="left", fill="x", expand=True)

        ttk.Label(self.left_frame, text="Output Folder").grid(row=12, column=0, sticky="w")
        self.output_path = tk.StringVar()
        output_frame = ttk.Frame(self.left_frame)
        output_frame.grid(row=13, column=0, sticky="ew", pady=(0, 15))
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        self.output_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side="right", padx=5)

        # --- Right side ---
        ttk.Label(self.right_frame, text="Settings").grid(row=0, column=0, sticky="w")

        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.frame_step = tk.IntVar(value=1)
        self.merge_iou = tk.DoubleVar(value=0.6)
        self.use_gpu = tk.BooleanVar(value=True)

        settings = [
            ("Confidence Threshold", self.conf_threshold, 0.01, 1.0, 0.01),
            ("IoU Threshold", self.iou_threshold, 0.01, 1.0, 0.01),
            ("Frame Step", self.frame_step, 1, 30, 1),
            ("Box Merge IoU", self.merge_iou, 0.01, 1.0, 0.01),
        ]

        for i, (label, var, a, b, step) in enumerate(settings, start=1):
            ttk.Label(self.right_frame, text=label).grid(row=i, column=0, sticky="w")

            if isinstance(var, tk.DoubleVar):
                scale_frame = ttk.Frame(self.right_frame)
                scale_frame.grid(row=i, column=1, sticky="ew", padx=(10, 0))

                # The scale itself
                scale = ttk.Scale(scale_frame, variable=var, from_=a, to=b, orient="horizontal", length=150)
                scale.pack(side="left", fill="x", expand=True)

                # Value label that updates live
                val_label = ttk.Label(scale_frame, text=f"{var.get():.2f}", width=5)
                val_label.pack(side="right", padx=(8, 0))

                # Update label as slider moves
                def update_label(val, lbl=val_label):
                    lbl.config(text=f"{float(val):.2f}")
                scale.config(command=update_label)

            else:
                ttk.Spinbox(self.right_frame, from_=a, to=b, textvariable=var, width=5).grid(
                    row=i, column=1, sticky="ew", padx=(10, 0)
                )


        ttk.Checkbutton(
            self.right_frame,
            text="Use GPU (fallback to CPU if unavailable)",
            variable=self.use_gpu,
        ).grid(row=6, column=0, columnspan=2, pady=(15, 10))

        ttk.Button(self.right_frame, text="Save Config", command=self.save_config).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(5, 5)
        )
        # Run Detection Button
        ttk.Button(self.right_frame, text="Run Detection", command=self.run_detection).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(5, 10)
        )

        # Log Output Label + Box
        ttk.Label(self.right_frame, text="Log Output").grid(row=9, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.log_text = tk.Text(self.right_frame, height=8, wrap="word")
        self.log_text.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(0, 5))
        self.log_text.configure(bg="#1e1e1e", fg="#f0f0f0", insertbackground="#f0f0f0", selectbackground="#3a3a3a")

        # Status Indicator (moved below log box)
        self.status_label = ttk.Label(
            self.right_frame,
            text="🟢 Ready",
            foreground="#9FEF9F",  # light green
            font=("Segoe UI", 10, "bold")
        )
        self.status_label.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(5, 5))
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
        self.eta_label.grid(row=12, column=0, columnspan=2, sticky="ew")

        # Cancel Button
        self.cancel_flag = False
        self.cancel_button = ttk.Button(self.right_frame, text="Cancel Detection", command=self.cancel_detection)
        self.cancel_button.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(5, 15))
        self.cancel_button.grid_remove()



        # Let the log box expand properly
        self.right_frame.rowconfigure(10, weight=1)

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
            "out_res": self.out_res.get() if self.out_res.get().lower() != "off" else None,  # 🆕 Add this
            "models": models,
            "out": self.output_path.get() or "./output",
            "conf": self.conf_threshold.get(),
            "iou": self.iou_threshold.get(),
            "frame_step": self.frame_step.get(),
            "merge_iou": self.merge_iou.get(),
            "cpu": not self.use_gpu.get(),
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
