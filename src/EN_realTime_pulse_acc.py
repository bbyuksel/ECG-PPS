# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:41:39 2024

@author: BBYCLUB_2022
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:01:08 2024

@author: BBYCLUB_2022
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from datetime import datetime

# Serial port settings
port = 'COM5'  # Write your serial port here
baud_rate = 9600

# Open serial port
ser = serial.Serial(port, baud_rate, timeout=1)

# List to store ECG data
ecg_data = []
timestamps = []
true_labels = []  # True labels are stored here
predicted_labels = []

# Load and compile the model
model = load_model('model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Matplotlib settings
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
line, = ax[0].plot([], [])
ax[0].set_ylim(-128, 128)
ax[0].set_title('Raw ECG Data')
line_filtered, = ax[1].plot([], [])
ax[1].set_ylim(-128, 128)
ax[1].set_title('Filtered ECG Data')

# Window size
window_size = 180

# Class labels
class_labels = {0: 'N', 1: 'L', 2: 'R', 3: 'A', 4: 'V'}

# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Function to start the module and read data
def read_ecg_data():
    global ecg_data, timestamps, pulse_value, predicted_class

    pulse_value = None
    fs = 500  # Sampling frequency
    lowcut = 0.5  # Lower cutoff frequency
    highcut = 50.0  # Upper cutoff frequency

    processed_data_count = 0
    start_time = time.time()

    while True:
        if ser.in_waiting > 0:
            data = ser.read(1)
            if data == b'\xF8':  # Wave sampling point marker
                wave_data = ser.read(300)  # Read 50 samples at 50 Hz
                wave_data = list(map(lambda x: x - 128, wave_data))  # Centering the ECG signal
                ecg_data.extend(wave_data)
                timestamps.extend([datetime.now()] * len(wave_data))

                if len(ecg_data) > 3000:  # Keep only the last 3000 data points
                    ecg_data = ecg_data[-3000:]
                    timestamps = timestamps[-3000:]

                # Use filtered or unfiltered data
                root.after(0, update_plot, ecg_data[-window_size:], timestamps[-window_size:], fs, lowcut, highcut)

                # Normalize and send data to the model
                if len(ecg_data) >= window_size:  # Send to model if there is enough data
                    test_data = (np.array(ecg_data[-window_size:]) - np.mean(ecg_data[-window_size:])) / np.std(ecg_data[-window_size:])
                    test_data = test_data.reshape(1, window_size, 1)  # Reshape to the expected format of the model

                    prediction_start_time = time.time()
                    prediction = model.predict(test_data)
                    prediction_end_time = time.time()
                    predicted_class = class_labels[np.argmax(prediction, axis=1)[0]]
                    predicted_labels.append(np.argmax(prediction, axis=1)[0])
                    true_labels.append(get_true_label())  # Get the true label from here

                    print("Predicted class:", predicted_class)

                    # Show predicted class on the interface
                    root.after(0, update_predictions, f'Predicted class: {predicted_class}')
                    
                    # Processing latency
                    latency = prediction_end_time - prediction_start_time
                    print(f'Processing latency: {latency:.4f} seconds')
                    root.after(0, update_latency, f'Latency: {latency:.4f} seconds')

                processed_data_count += 1
                if time.time() - start_time >= 1:
                    root.after(0, update_processed_data_count, f'Processed data count: {processed_data_count} data/second')
                    processed_data_count = 0
                    start_time = time.time()

            elif data == b'\xFA':  # Pulse value marker
                pulse_data = ser.read(1)
                pulse_value = ord(pulse_data)
                print("Pulse value:", pulse_value)
                # Show pulse value on the interface
                root.after(0, update_pulse, f'Pulse value: {pulse_value}')

            elif data == b'\xFB':  # Info byte marker
                info_data = ser.read(1)
                if info_data == b'\x11':
                    print("Lead off detected")
                else:
                    print("Unknown info byte:", ord(info_data))
            else:
                print("Unknown data byte:", ord(data))
        time.sleep(0.01)  # Short delay to reduce CPU usage

def update_plot(ecg_segment, time_segment, fs, lowcut, highcut):
    if filter_var.get():
        data_to_plot = butter_bandpass_filter(np.array(ecg_segment), lowcut, highcut, fs, order=4)
    else:
        data_to_plot = np.array(ecg_segment)

    time_labels = [time.strftime("%H:%M:%S") for time in time_segment]

    line.set_ydata(ecg_segment)
    line.set_xdata(range(len(ecg_segment)))
    ax[0].set_xlim(0, len(ecg_segment))
    ax[0].set_xticks(range(0, len(ecg_segment), fs))  # Mark every 1 second
    ax[0].set_xticklabels(time_labels[::fs])  # Add timestamp every 1 second
    ax[0].set_ylim(min(ecg_segment), max(ecg_segment))
    fig.canvas.draw()
    fig.canvas.flush_events()

    line_filtered.set_ydata(data_to_plot)
    line_filtered.set_xdata(range(len(data_to_plot)))
    ax[1].set_xlim(0, len(data_to_plot))
    ax[1].set_xticks(range(0, len(data_to_plot), fs))  # Mark every 1 second
    ax[1].set_xticklabels(time_labels[::fs])  # Add timestamp every 1 second
    ax[1].set_ylim(min(data_to_plot), max(data_to_plot))
    fig.canvas.draw()
    fig.canvas.flush_events()

def start_reading():
    threading.Thread(target=read_ecg_data, daemon=True).start()

def update_predictions(text):
    prediction_label.config(text=text)

def update_pulse(text):
    pulse_label.config(text=text)

def update_latency(text):
    latency_label.config(text=text)

def update_processed_data_count(text):
    data_count_label.config(text=text)

def update_performance_metrics():
    if true_labels and predicted_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
        accuracy_label.config(text=f'Accuracy: {accuracy:.2%}')
        
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels.values())
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

# Add a function to get the true label. This will vary depending on how true data is collected.
def get_true_label():
    # This is an example of the function that will get the true label. Adapt to your actual data source.
    # For example, it could be manual labeling or labels from another data source.
    return np.random.randint(0, 5)  # Return a random label as an example.

# Create Tkinter interface
root = tk.Tk()
root.title("Real-Time ECG Monitoring and Disease Diagnosis")

main_frame = ttk.Frame(root)
main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

plot_frame = ttk.Frame(main_frame)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

control_frame = ttk.Frame(main_frame)
control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

# Add Checkbutton
filter_var = tk.BooleanVar()
filter_checkbox = ttk.Checkbutton(control_frame, text="Apply Filter", variable=filter_var)
filter_checkbox.pack(pady=10)

# Add Start button
start_button = ttk.Button(control_frame, text="Start Reading", command=start_reading)
start_button.pack(pady=10)

# Predicted class, pulse value, processing latency, processed data count, and accuracy labels
prediction_label = ttk.Label(control_frame, text="Predicted class: ", font=("Arial", 14))
prediction_label.pack(pady=10)
pulse_label = ttk.Label(control_frame, text="Pulse value: ", font=("Arial", 14))
pulse_label.pack(pady=10)
latency_label = ttk.Label(control_frame, text="Latency: ", font=("Arial", 14))
latency_label.pack(pady=10)
data_count_label = ttk.Label(control_frame, text="Processed data count: ", font=("Arial", 14))
data_count_label.pack(pady=10)
accuracy_label = ttk.Label(control_frame, text="Accuracy: ", font=("Arial", 14))
accuracy_label.pack(pady=10)

# Add Matplotlib plot to Tkinter window
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Add Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Start timer to update performance metrics
def periodic_performance_update():
    update_performance_metrics()
    root.after(1000, periodic_performance_update)  # Update every 1 second

periodic_performance_update()

# Tkinter main loop
root.mainloop()

# Close serial port
ser.close()

# Comments:
# Processing Latency: The model's prediction time is measured using prediction_start_time and prediction_end_time and printed on the screen.
# Data Processing Rate: The amount of data processed per second is measured using processed_data_count and printed on the screen.
# Accuracy: The accuracy rate is calculated using true_labels and predicted_labels lists and printed on the screen.
# Confusion Matrix: A confusion matrix is calculated and displayed using true and predicted classes.
# Visualization: Raw and filtered ECG signals are displayed as graphs using Matplotlib.
# How to Obtain True Labels in Real-Time Applications?
# Pre-labeled Data: A pre-labeled data set can be loaded into the system and used in parallel with real-time data.
# User Input: The user can manually label the data in real-time. This can be done by having the user identify the correct classes in the application.
# Reference System: A reliable reference system (e.g., expert-verified data or another reliable algorithm) can be used to determine the true labels.
