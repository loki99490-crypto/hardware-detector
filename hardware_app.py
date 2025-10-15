import os
import tkinter as tk
from tkinter import filedialog, StringVar, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ---------- CONFIG ----------
MODEL_PATH = "hardware_model.h5"
DATASET_PATH = "hardware_dataset/"
IMG_SIZE = (224, 224)

BG_START = "g_start.jpg"
BG_MENU = "bg_menu.jpg"
BG_DETECTION = "bg_detection.jpg"

# ---------- MODEL LOADING ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model '{MODEL_PATH}' not found.")
model = load_model(MODEL_PATH)

class_indices = {i: f for i, f in enumerate(
    sorted(d for d in os.listdir(DATASET_PATH)
           if os.path.isdir(os.path.join(DATASET_PATH, d)))
)}

# ---------- FUNCTIONS ----------
def predict_hardware(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return class_indices[idx], preds[idx] * 100

def get_sample_image_for_component(component):
    folder = os.path.join(DATASET_PATH, component)
    if not os.path.exists(folder):
        return None
    imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    return os.path.join(folder, random.choice(imgs)) if imgs else None

def display_result(img_path):
    switch_frame(detect_frame)
    component, accuracy = predict_hardware(img_path)
    info_text.set(f"Detected: {component}\nAccuracy: {accuracy:.2f}%")

    img_input = Image.open(img_path).resize((640, 480))
    img_input_tk = ImageTk.PhotoImage(img_input)
    panel_input.config(image=img_input_tk)
    panel_input.image = img_input_tk

    sample_path = get_sample_image_for_component(component)
    if sample_path:
        img_sample = Image.open(sample_path).resize((640, 480))
        img_sample_tk = ImageTk.PhotoImage(img_sample)
        panel_sample.config(image=img_sample_tk)
        panel_sample.image = img_sample_tk
    else:
        panel_sample.config(image='', text="No sample image found")

def upload_image():
    path = filedialog.askopenfilename()
    if path:
        display_result(path)

def capture_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        info_text.set("Camera not found")
        return

    info_text.set("Press SPACE to capture, ESC to cancel")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("IRON MAN Scanner", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 32:
            cv2.imwrite("captured.jpg", frame)
            display_result("captured.jpg")
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- GUI SETUP ----------
root = tk.Tk()
root.title("IRON MAN Hardware Analyzer")
root.geometry("1500x900")

def set_background(frame, img_path):
    if os.path.exists(img_path):
        bg = Image.open(img_path).resize((1500, 900))
        bg_img = ImageTk.PhotoImage(bg)
        lbl_bg = Label(frame, image=bg_img)
        lbl_bg.image = bg_img
        lbl_bg.place(x=0, y=0, relwidth=1, relheight=1)

def switch_frame(target_frame):
    for f in [start_frame, menu_frame, detect_frame]:
        f.pack_forget()
    target_frame.pack(fill="both", expand=True)

button_style = {
    "bg": "#1a1a1a",
    "fg": "#00eaff",
    "activebackground": "#ff0000",
    "activeforeground": "white",
    "font": ("Consolas", 14, "bold"),
    "relief": "ridge",
    "bd": 3
}

label_style = {
    "bg": "#000000",
    "fg": "#00eaff",
    "font": ("Courier New", 13, "bold")
}

# ---------- START SCREEN ----------
start_frame = tk.Frame(root)
set_background(start_frame, BG_START)

heading = Label(start_frame, text="COMPUTER HARDWARE DETECTION", 
                fg="#00eaff", bg="#000000", font=("Consolas", 28, "bold"))
heading.pack(pady=80)

Button(start_frame, text="START", command=lambda: switch_frame(menu_frame), **button_style).pack(expand=True)

# ---------- MENU SCREEN ----------
menu_frame = tk.Frame(root)
set_background(menu_frame, BG_MENU)
Button(menu_frame, text="📁 Upload Image", command=upload_image, **button_style).pack(pady=20)
Button(menu_frame, text="📷 Launch Scanner", command=capture_camera, **button_style).pack(pady=20)
Button(menu_frame, text="⬅ Back", command=lambda: switch_frame(start_frame), **button_style).pack(pady=40)

# ---------- DETECTION SCREEN ----------
detect_frame = tk.Frame(root)
set_background(detect_frame, BG_DETECTION)

frame_images = tk.Frame(detect_frame, bg="#000")
frame_images.pack(pady=20)

# Input and Reference panels
panel_input = Label(frame_images, width=640, height=480, bg="#000")
panel_input.pack(side="left", padx=10)

panel_sample = Label(frame_images, width=640, height=480, bg="#000")
panel_sample.pack(side="right", padx=10)

# Image labels below each panel
labels_frame = tk.Frame(detect_frame, bg="#000")
labels_frame.pack()

Label(labels_frame, text="Input Image", **label_style).pack(side="left", padx=300)
Label(labels_frame, text="Reference Image", **label_style).pack(side="right", padx=280)

# Detection result text
info_text = StringVar()
info_label = Label(detect_frame, textvariable=info_text, **label_style)
info_label.pack(pady=20)

Button(detect_frame, text="🏠 Home", command=lambda: switch_frame(menu_frame), **button_style).pack(pady=10)

# ---------- INIT ----------
switch_frame(start_frame)
root.mainloop()
