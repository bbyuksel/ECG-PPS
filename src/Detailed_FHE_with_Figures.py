# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 00:04:07 2024

@author: BBYCLUB_2022
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:29:45 2024

@author: BBYCLUB_2022
"""

import serial
import tkinter as tk
from tkinter import ttk
import threading
import tenseal as ts
import time
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Serial port and data variables
ser = serial.Serial('COM5', 9600, timeout=1)
ecg_data = []
window_size = 3000
sampling_rate = 300  # Sample rate in Hz

# Create data queue
stored_data = []
encrypted_data = []
decrypted_data = []
is_reading = False

context = None  # Context object is defined globally
secret_key = None  # Secret key is defined globally

# Bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Derivative
def derivative(data):
    return np.diff(data)

# Squaring
def squaring(data):
    return data ** 2

# Moving window integration
def moving_window_integration(data, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

# Detect R-peaks in ECG signal
def pan_tompkins(ecg, fs):
    # Bandpass filter
    filtered_ecg = bandpass_filter(ecg, 5, 15, fs)
    
    # Derivative
    diff_ecg = derivative(filtered_ecg)
    
    # Squaring
    squared_ecg = squaring(diff_ecg)
    
    # Moving average
    mwa_ecg = moving_window_integration(squared_ecg, int(0.150 * fs))
    
    # Thresholding and R-peak detection
    peaks, _ = find_peaks(mwa_ecg, distance=int(0.2 * fs), height=np.mean(mwa_ecg))
    
    return peaks

# Start module and read data function
def read_ecg_data():
    global ecg_data, stored_data

    pulse_value = None

    while is_reading:
        if ser.in_waiting > 0:
            data = ser.read(1)
            if data == b'\xF8':  # Wave sampling point marker
                wave_data = ser.read(50)  # Read 50 samples at 50 Hz
                wave_data = list(map(lambda x: x - 128, wave_data))  # Center the ECG signal
                ecg_data.extend(wave_data)
                stored_data.extend(wave_data)  # Save the data
                if len(ecg_data) > 3000:  # Keep only the last 3000 data
                    ecg_data = ecg_data[-3000:]
                print("Wave data:", wave_data)
            elif data == b'\xFA':  # Pulse value marker
                pulse_data = ser.read(1)
                pulse_value = ord(pulse_data)
                print("Pulse value:", pulse_value)
            elif data == b'\xFB':  # Information byte marker
                info_data = ser.read(1)
                if info_data == b'\x11':
                    print("Lead off detected")
                else:
                    print("Unknown info byte:", ord(info_data))
            else:
                print("Unknown data byte:", ord(data))
        time.sleep(0.01)  # Short delay to reduce CPU usage

    ser.close()  # Close the serial port

def start_reading():
    global is_reading
    is_reading = True
    threading.Thread(target=read_ecg_data).start()
    print("Started reading")

def stop_reading():
    global is_reading
    is_reading = False
    print("Stopped reading, data saved")

def encrypt_data(data):
    global encrypted_data, context, secret_key
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    secret_key = context.secret_key()  # Store the secret key
    public_context = ts.context_from(context.serialize(save_secret_key=False))
    encrypted_data = [ts.ckks_vector(public_context, [val]) for val in data]
    print("Data encrypted")
    return context

def decrypt_data(context):
    global decrypted_data
    decrypted_data = [vec.decrypt(secret_key=secret_key) for vec in encrypted_data]
    decrypted_data = [item[0] for item in decrypted_data]
    print("Data decrypted")
    return decrypted_data

def calculate_accuracy(original_data, decrypted_data):
    if len(original_data) != len(decrypted_data):
        raise ValueError("The length of original data and decrypted data must be the same")
    original_data = np.array(original_data)
    decrypted_data = np.array(decrypted_data)
    accuracy = np.mean(np.isclose(original_data, decrypted_data, rtol=1e-5))
    return accuracy

def start_fhe_encrypt():
    global context
    context = encrypt_data(stored_data)

def start_fhe_decrypt():
    global context
    decrypted_data = decrypt_data(context)
    accuracy = calculate_accuracy(stored_data, decrypted_data)
    accuracy_label.config(text=f'Accuracy: {accuracy:.4f}')
    plot_decrypted_data(decrypted_data)  # Plot decrypted data

# Run Pan and Tompkins algorithm on encrypted data
def analyze_encrypted_data():
    global context, encrypted_data, secret_key
    encrypted_data_np = np.array([vec.decrypt(secret_key=secret_key)[0] for vec in encrypted_data])
    encrypted_r_peaks = pan_tompkins(encrypted_data_np, sampling_rate)  # Assume 500 sampling frequency
    print(f'Encrypted R-peaks: {encrypted_r_peaks}')
    return encrypted_r_peaks

# Basic Statistical Analysis
def statistical_analysis_encrypted():
    global encrypted_data, secret_key
    # Decrypt the encrypted data
    decrypted_data = np.array([vec.decrypt(secret_key=secret_key)[0] for vec in encrypted_data])
    # Basic statistical analysis
    mean_value = np.mean(decrypted_data)
    std_dev = np.std(decrypted_data)
    median_value = np.median(decrypted_data)
    min_value = np.min(decrypted_data)
    max_value = np.max(decrypted_data)
    print(f'Mean: {mean_value}, Std Dev: {std_dev}, Median: {median_value}, Min: {min_value}, Max: {max_value}')
    return mean_value, std_dev, median_value, min_value, max_value

# Frequency Analysis
def frequency_analysis_encrypted():
    global encrypted_data, secret_key
    # Decrypt the encrypted data
    decrypted_data = np.array([vec.decrypt(secret_key=secret_key)[0] for vec in encrypted_data])
    # Frequency analysis using Fourier transform
    fft_data = np.fft.fft(decrypted_data)
    freqs = np.fft.fftfreq(len(decrypted_data))
    print(f'FFT Frequencies: {freqs}')
    print(f'FFT Data: {fft_data}')
    return freqs, fft_data

# Heart Rate Variability (HRV) Analysis
def hrv_analysis_encrypted():
    global encrypted_data, secret_key
    # Decrypt the encrypted data
    decrypted_data = np.array([vec.decrypt(secret_key=secret_key)[0] for vec in encrypted_data])
    # Detect R-peaks
    r_peaks = pan_tompkins(decrypted_data, sampling_rate)
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate  # RR intervals in seconds
    # HRV analysis
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    print(f'Mean RR Interval: {mean_rr}, Std Dev RR Interval: {std_rr}')
    return mean_rr, std_rr

# Plotting functions
def plot_raw_and_filtered_data():
    time_axis = np.arange(0, len(ecg_data[:window_size])) / sampling_rate
    
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, ecg_data[:window_size], label='Raw ECG Data')
    plt.title('Raw ECG Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    filtered_ecg = bandpass_filter(np.array(ecg_data[:window_size]), 5, 15, sampling_rate)
    plt.plot(time_axis, filtered_ecg, label='Filtered ECG Data', color='r')
    plt.title('Filtered ECG Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    encrypted_ecg = [vec.decrypt(secret_key=secret_key)[0] for vec in encrypted_data[:window_size]]
    plt.plot(time_axis, encrypted_ecg, label='Encrypted ECG Data', color='g')
    plt.title('Encrypted ECG Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_decrypted_data(decrypted_data):
    time_axis = np.arange(0, len(stored_data[:window_size])) / sampling_rate

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, stored_data[:window_size], label='Original Data')
    plt.plot(time_axis, decrypted_data[:window_size], label='Decrypted Data', linestyle='--')
    plt.title('Original and Decrypted ECG Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def plot_frequency_analysis():
    freqs, fft_data = frequency_analysis_encrypted()
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, np.abs(fft_data))
    plt.title('Frequency Analysis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

def on_closing():
    global is_reading
    is_reading = False
    ser.close()
    root.destroy()

# Create Tkinter interface
root = tk.Tk()
root.title("Serial Port Data Reading and Encryption")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Button frame
button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

# Buttons
start_button = ttk.Button(button_frame, text="Start", command=start_reading)
start_button.grid(row=0, column=0, padx=5, pady=5)

stop_button = ttk.Button(button_frame, text="Stop", command=stop_reading)
stop_button.grid(row=0, column=1, padx=5, pady=5)

encrypt_button = ttk.Button(button_frame, text="FHE Encrypt", command=start_fhe_encrypt)
encrypt_button.grid(row=0, column=2, padx=5, pady=5)

decrypt_button = ttk.Button(button_frame, text="FHE Decrypt", command=start_fhe_decrypt)
decrypt_button.grid(row=0, column=3, padx=5, pady=5)

analyze_button = ttk.Button(button_frame, text="Analyze Encrypted Data", command=analyze_encrypted_data)
analyze_button.grid(row=0, column=4, padx=5, pady=5)

stat_analysis_button = ttk.Button(button_frame, text="Statistical Analysis", command=statistical_analysis_encrypted)
stat_analysis_button.grid(row=0, column=5, padx=5, pady=5)

freq_analysis_button = ttk.Button(button_frame, text="Frequency Analysis", command=plot_frequency_analysis)
freq_analysis_button.grid(row=0, column=6, padx=5, pady=5)

hrv_analysis_button = ttk.Button(button_frame, text="HRV Analysis", command=hrv_analysis_encrypted)
hrv_analysis_button.grid(row=0, column=7, padx=5, pady=5)

plot_button = ttk.Button(button_frame, text="Plot Raw & Filtered", command=plot_raw_and_filtered_data)
plot_button.grid(row=0, column=8, padx=5, pady=5)

exit_button = ttk.Button(button_frame, text="Exit", command=on_closing)
exit_button.grid(row=0, column=9, padx=5, pady=5)

# Accuracy label
accuracy_label = ttk.Label(root, text="Accuracy: N/A")
accuracy_label.pack(side=tk.BOTTOM)

# Tkinter main loop
root.mainloop()

# Close serial port
ser.close()
