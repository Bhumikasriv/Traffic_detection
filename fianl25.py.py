import threading
import cv2
from collections import Counter
from ultralytics import YOLO
import serial
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import joblib
from speechbrain.pretrained import EncoderClassifier
import os

# Global variables
global pred
pred = 0

# -------------- Speaker Recognition Setup --------------
classifier_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def extract_embedding(file_path):
    signal = classifier_model.load_audio(file_path)
    embedding = classifier_model.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()

def extract_embedding_array(audio_data, sample_rate):
    tmp_file = "temp_audio.wav"
    wav.write(tmp_file, sample_rate, audio_data.astype(np.int16))
    emb = extract_embedding(tmp_file)
    os.remove(tmp_file)
    return emb

MODEL_PATH = "speaker_model_embeddings.pkl"
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", "Trained model not found! Please train first.")
    raise SystemExit

clf = joblib.load(MODEL_PATH)
speakers = clf.classes_

# -------------- Arduino Setup --------------
arduino_port = 'COM90'  # Change as needed
baud_rate = 9600
try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
except Exception as e:
    arduino = None
    print(f"Arduino connection error: {e}")

# -------------- YOLO Setup --------------
model = YOLO('best.pt')
class_names = ['car']
colors = {0: (0, 255, 0)}

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

confidence_threshold = 0.4

def yolo_detection_loop():
    last_send_time = 0
    send_interval = 5  # Send data every 5 seconds
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not ret_left or not ret_right:
            print("Error: Camera stream disconnected.")
            break

        # Process left camera
        results_left = model(frame_left, conf=confidence_threshold, verbose=False)
        detected_objects_left = []
        for result in results_left:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detected_objects_left.append(cls)
                color = colors.get(cls, (0, 255, 0))
                cv2.rectangle(frame_left, (x1, y1), (x2, y2), color, 2)
                label_text = f'{class_names[cls]}: {conf:.2f}'
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_left, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame_left, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        object_counts_left = Counter(detected_objects_left)
        cv2.putText(frame_left, "LEFT SIDE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Process right camera
        results_right = model(frame_right, conf=confidence_threshold, verbose=False)
        detected_objects_right = []
        for result in results_right:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detected_objects_right.append(cls)
                color = colors.get(cls, (0, 255, 0))
                cv2.rectangle(frame_right, (x1, y1), (x2, y2), color, 2)
                label_text = f'{class_names[cls]}: {conf:.2f}'
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_right, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame_right, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        object_counts_right = Counter(detected_objects_right)
        cv2.putText(frame_right, "RIGHT SIDE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        left_count = object_counts_left.get(0, 0)
        right_count = object_counts_right.get(0, 0)
        print(f"Left Side Cars: {left_count} | Right Side Cars: {right_count}")

        # Send to Arduino only every 5 seconds
        current_time = time.time()
        if arduino and arduino.is_open and (current_time - last_send_time) >= send_interval:
            message = f"{left_count}:{right_count}:{pred}:\n"
            print(f"Sending to Arduino: {message.strip()}")
            try:
                arduino.write(message.encode())
                last_send_time = current_time  # Update last send time
            except Exception as e:
                print(f"Failed to send to Arduino: {e}")

        cv2.imshow('Camera LEFT', frame_left)
        cv2.imshow('Camera RIGHT', frame_right)

        # Use small waitKey with check for quit to keep GUI responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
        print("Arduino connection closed")

# -------------- Enhanced Tkinter GUI Setup --------------
class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Speaker Recognition System")
        self.root.geometry("700x550")
        self.root.resizable(True, True)
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Setup GUI
        self.setup_gui()
        
        # Recording state
        self.is_recording = False
        self.recording = None
        
    def configure_styles(self):
        self.style.configure('Title.TLabel', 
                           font=('Arial', 22, 'bold'), 
                           background='#f0f0f0',
                           foreground='#2c3e50')
        
        self.style.configure('Subtitle.TLabel',
                           font=('Arial', 12),
                           background='#f0f0f0',
                           foreground='#7f8c8d')
        
        self.style.configure('Card.TFrame',
                           background='white',
                           relief='raised',
                           borderwidth=1)
        
        self.style.configure('Primary.TButton',
                           font=('Arial', 12, 'bold'),
                           padding=(20, 10),
                           background='#3498db',
                           foreground='white')
        
        self.style.configure('Success.TButton',
                           font=('Arial', 12, 'bold'),
                           padding=(20, 10),
                           background='#2ecc71',
                           foreground='white')
        
        self.style.configure('Warning.TButton',
                           font=('Arial', 12, 'bold'),
                           padding=(20, 10),
                           background='#e74c3c',
                           foreground='white')
        
        self.style.configure('Result.TLabel',
                           font=('Arial', 13, 'bold'),
                           background='white',
                           foreground='#2c3e50',
                           justify='center')
        
        self.style.map('Primary.TButton',
                      background=[('active', '#2980b9')])
        self.style.map('Success.TButton',
                      background=[('active', '#27ae60')])
        self.style.map('Warning.TButton',
                      background=[('active', '#c0392b')])
    
    def setup_gui(self):
        # Header
        header_frame = ttk.Frame(self.root, style='Card.TFrame')
        header_frame.pack(fill='x', padx=20, pady=20)
        
        title_label = ttk.Label(header_frame, 
                              text="Intelligent Speaker Recognition", 
                              style='Title.TLabel')
        title_label.pack(pady=(15, 5))
        
        subtitle_label = ttk.Label(header_frame, 
                                 text="AI-Powered Voice Identification and Real-time Object Detection", 
                                 style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 15))
        
        # Main content frame
        main_frame = ttk.Frame(self.root, style='Card.TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Input methods section
        input_frame = ttk.LabelFrame(main_frame, text="Input Methods", padding=20)
        input_frame.pack(fill='x', padx=15, pady=15)
        
        # File selection button
        self.select_button = ttk.Button(input_frame, 
                                      text="📁 Select Audio File", 
                                      style='Primary.TButton',
                                      command=self.select_audio)
        self.select_button.pack(fill='x', pady=10)
        
        # Recording button
        self.record_button = ttk.Button(input_frame, 
                                      text="🎤 Record from Microphone", 
                                      style='Success.TButton',
                                      command=self.toggle_recording)
        self.record_button.pack(fill='x', pady=10)
        
        # Recording progress
        self.progress = ttk.Progressbar(input_frame, mode='indeterminate')
        
        # Results section
        result_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding=20)
        result_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.result_label = ttk.Label(result_frame, 
                                    text="Please select an audio file or start recording", 
                                    style='Result.TLabel',
                                    wraplength=500)
        self.result_label.pack(fill='both', expand=True, pady=20)
        
        # Confidence section
        confidence_frame = ttk.LabelFrame(main_frame, text="Confidence Levels", padding=15)
        confidence_frame.pack(fill='x', padx=15, pady=10)
        
        self.confidence_text = tk.Text(confidence_frame, 
                                     height=4, 
                                     font=('Arial', 10),
                                     bg='#f8f9fa',
                                     relief='flat')
        self.confidence_text.pack(fill='x')
        self.confidence_text.insert('1.0', 'Confidence details will appear here...')
        self.confidence_text.config(state='disabled')
        
        # Status bar
        status_frame = ttk.Frame(self.root, style='Card.TFrame')
        status_frame.pack(fill='x', padx=20, pady=10)
        
        self.status_label = ttk.Label(status_frame, 
                                    text="System Ready | YOLO Detection Running | Serial: 5s Interval", 
                                    style='Subtitle.TLabel')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Exit button
        exit_button = ttk.Button(status_frame, 
                               text="Exit", 
                               style='Warning.TButton',
                               command=self.root.destroy)
        exit_button.pack(side='right', padx=10, pady=5)
    
    def select_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.ogg *.flac *.mp3"), 
                      ("All Files", "*.*")]
        )
        if file_path:
            self.status_label.config(text="Processing audio file...")
            self.root.update()
            self.predict_from_file(file_path)
    
    def predict_from_file(self, file_path):
        global pred
        try:
            emb = extract_embedding(file_path)
            pred = clf.predict([emb])[0]
            conf = clf.predict_proba([emb])[0]
            conf_dict = {sp: round(float(c)*100, 2) for sp, c in zip(speakers, conf)}
            
            # Update result label
            result_text = f"🎯 Predicted Speaker: {pred}\n📊 Confidence: {conf_dict[pred]}%"
            self.result_label.config(text=result_text, foreground='#27ae60')
            
            # Update confidence details
            self.update_confidence_display(conf_dict)
            
            self.status_label.config(text="Audio file processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process audio:\n{str(e)}")
            self.status_label.config(text="Error processing audio file")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                messagebox.showerror("Error", "No input (microphone) device found!")
                return
            
            self.is_recording = True
            self.record_button.config(text="⏹️ Stop Recording", style='Warning.TButton')
            self.result_label.config(text="🎤 Recording... Speak now!", foreground='#e67e22')
            self.progress.pack(fill='x', pady=10)
            self.progress.start()
            self.status_label.config(text="Recording in progress...")
            
            self.recording = None
            self.sample_rate = 16000
            self.duration = 4
            
            # Start recording in a separate thread to keep GUI responsive
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording:\n{str(e)}")
            self.is_recording = False
    
    def record_audio(self):
        try:
            self.recording = sd.rec(int(self.duration * self.sample_rate), 
                                  samplerate=self.sample_rate, 
                                  channels=1, 
                                  dtype='int16')
            sd.wait()
            
            # Schedule the processing in the main thread
            self.root.after(0, self.process_recording)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed:\n{str(e)}"))
            self.root.after(0, self.reset_recording_ui)
    
    def process_recording(self):
        global pred
        try:
            audio_np = self.recording.flatten()
            self.result_label.config(text="🔍 Processing audio...", foreground='#f39c12')
            self.status_label.config(text="Processing recorded audio...")
            self.root.update()
            
            emb = extract_embedding_array(audio_np, self.sample_rate)
            pred = clf.predict([emb])[0]
            conf = clf.predict_proba([emb])[0]
            conf_dict = {sp: round(float(c)*100, 2) for sp, c in zip(speakers, conf)}
            
            result_text = f"🎯 Predicted Speaker: {pred}\n📊 Confidence: {conf_dict[pred]}%"
            self.result_label.config(text=result_text, foreground='#27ae60')
            self.update_confidence_display(conf_dict)
            self.status_label.config(text="Recording processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process recording:\n{str(e)}")
            self.status_label.config(text="Error processing recording")
        
        self.reset_recording_ui()
    
    def stop_recording(self):
        self.is_recording = False
        sd.stop()
        self.reset_recording_ui()
        self.result_label.config(text="Recording stopped", foreground='#e74c3c')
        self.status_label.config(text="Recording stopped by user")
    
    def reset_recording_ui(self):
        self.is_recording = False
        self.record_button.config(text="🎤 Record from Microphone", style='Success.TButton')
        self.progress.stop()
        self.progress.pack_forget()
    
    def update_confidence_display(self, conf_dict):
        self.confidence_text.config(state='normal')
        self.confidence_text.delete('1.0', 'end')
        
        # Sort by confidence
        sorted_conf = sorted(conf_dict.items(), key=lambda x: x[1], reverse=True)
        
        for speaker, confidence in sorted_conf:
            color = "#27ae60" if confidence > 70 else "#f39c12" if confidence > 40 else "#e74c3c"
            self.confidence_text.insert('end', f"• {speaker}: ", 'normal')
            self.confidence_text.insert('end', f"{confidence}%\n", ('color', color))
        
        self.confidence_text.config(state='disabled')
        
        # Configure tags for colors
        self.confidence_text.tag_configure('color', foreground=color)
        self.confidence_text.tag_configure('normal', foreground='#2c3e50')

# -------------- Run Application --------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernGUI(root)
    
    # -------------- Run YOLO detection in background thread --------------
    yolo_thread = threading.Thread(target=yolo_detection_loop, daemon=True)
    yolo_thread.start()
    
    root.mainloop()