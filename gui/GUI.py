import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading, sys
import steps  # import steps.py

class RedirectText:
    def __init__(self, text_widget): 
        self.output = text_widget
    def write(self, string):
        self.output.configure(state='normal')
        self.output.insert(tk.END, string)
        self.output.see(tk.END)
        self.output.configure(state='disabled')
    def flush(self): 
        pass

class BrainSliceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Slice Analysis Pipeline")
        self.root.geometry("800x600")

        # Left panel for buttons
        frame_left = tk.Frame(root)
        frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        steps_list = [
            ("1 Adjust brain slice orientation", steps.step1_main),
            ("2 Standardize image size", steps.step2_main),
            ("3 Extract fluorescence channels (NMF)", steps.step3_main),
            ("4 Align images (SimpleITK)", steps.step4_main),
            ("5 Downsample images", steps.step5_main),
            ("6 Normalize fluorescence intensity", steps.step6_main),
            ("7 Extract brain region coordinates", steps.step7_main),
        ]
        for label, func in steps_list:
            btn = ttk.Button(frame_left, text=label, command=lambda f=func:self.run_step(f))
            btn.pack(fill=tk.X, pady=2)

        # Right panel for log output
        frame_right = tk.Frame(root)
        frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.log = scrolledtext.ScrolledText(frame_right, state='disabled')
        self.log.pack(fill=tk.BOTH, expand=True)

        # Redirect print statements
        sys.stdout = RedirectText(self.log)
        sys.stderr = RedirectText(self.log)

    def run_step(self, func):
        def task():
            try: 
                func()
                messagebox.showinfo("Finished", "This step has been completed successfully.")
            except Exception as e: 
                messagebox.showerror("Error", f"Execution failed: {e}")

        # Step7 (Napari) must run in the main thread
        if func.__name__ == "step7_main":
            task()
        else:
            threading.Thread(target=task).start()

if __name__=="__main__":
    root = tk.Tk()
    app = BrainSliceGUI(root)
    root.mainloop()
