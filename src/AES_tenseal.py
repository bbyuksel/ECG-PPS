# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:48:29 2024

@author: BBYCLUB_2022
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import cryptography
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import os
import json
import serial
import time
import tenseal as ts

# Seri port ayarları
port = 'COM5'  # Seri portunuzu buraya yazın
baud_rate = 9600

# Seri portu aç
ser = serial.Serial(port, baud_rate, timeout=1)

# EKG verilerini saklamak için liste
ecg_data = []

# Modeli yükleyin ve derleyin
model = load_model('model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Matplotlib ayarları
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Filtered ECG Signal')
ax.set_ylim(-128, 128)
ax.set_xlim(0, 300)
text_pred = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')
text_pulse = ax.text(0.5, 1.1, '', transform=ax.transAxes, ha='center')

# P-R, Q-R, S-T periyotlarını ve P dalgası, QRS süresini saklamak için listeler
pr_intervals = []
qr_intervals = []
st_intervals = []
p_wave_durations = []
qrs_durations = []

# Pencere boyutu
window_size = 180

# Sınıf etiketleri
class_labels = {0: 'N', 1: 'L', 2: 'R', 3: 'A', 4: 'V'}

# AES şifreleme için key ve iv oluşturma
password = b"supersecret"  # Güvenli bir şekilde saklanmalı ve paylaşılmalıdır
salt = os.urandom(16)
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
    backend=default_backend()
)
key = kdf.derive(password)
iv = os.urandom(16)

def aes_encrypt(data):
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data

def aes_decrypt(encrypted_data):
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(decrypted_data) + unpadder.finalize()
    return data

# FHE şifreleme için TenSEAL kullanma
def fhe_encrypt(data):
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
    context.global_scale = 2**20
    context.generate_galois_keys()
    encrypted_vector = ts.bfv_vector(context, data)
    return encrypted_vector.serialize(), context

def fhe_decrypt(encrypted_vector, context):
    encrypted_vector = ts.bfv_vector_from(context, encrypted_vector)
    decrypted_data = encrypted_vector.decrypt()
    return decrypted_data

# Butterworth band geçiş filtresi
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

# Tepe noktalarını tespit etme fonksiyonu
def detect_peaks(ecg_signal, distance=150):
    peaks, _ = find_peaks(ecg_signal, distance=distance)
    return peaks

# EKG dalgalarını ve zaman periyotlarını belirleme fonksiyonu
def analyze_ecg(ecg_signal, fs):
    r_peaks = detect_peaks(ecg_signal, distance=int(fs*0.6))  # R dalgası tespiti
    q_peaks = detect_peaks(-ecg_signal, distance=int(fs*0.2))  # Q dalgası tespiti (negatif tepe noktası)
    s_peaks = detect_peaks(-ecg_signal, distance=int(fs*0.2))  # S dalgası tespiti (negatif tepe noktası)
    
    pr_intervals = np.diff(r_peaks) / fs * 1000  # P-R aralıkları
    qrs_durations = np.diff(q_peaks) / fs * 1000  # QRS süreleri
    qt_intervals = np.diff(s_peaks) / fs * 1000  # Q-T aralıkları
    
    return r_peaks, q_peaks, s_peaks, pr_intervals, qrs_durations, qt_intervals

# Modülü başlatma ve veri okuma fonksiyonu
def read_ecg_data():
    global ecg_data, pulse_value, predicted_class

    pulse_value = None
    fs = 500  # Örnekleme frekansı
    lowcut = 0.5  # Alt kesim frekansı
    highcut = 50.0  # Üst kesim frekansı

    while True:
        if ser.in_waiting > 0:
            data = ser.read(1)
            if data == b'\xF8':  # Dalga örnekleme noktası işareti
                wave_data = ser.read(50)  # 50 Hz hızında 50 örnek okur
                ecg_data.extend(list(wave_data))
                if len(ecg_data) > 3000:  # Sadece son 3000 veriyi tutar
                    ecg_data[:] = ecg_data[-3000:]

                # Filtreli veya filtresiz veri kullanımı
                root.after(0, update_plot, ecg_data[-window_size:], fs, lowcut, highcut)

                # Veriyi normalize edin ve modele gönderin
                if len(ecg_data) >= window_size:  # Yeterli veri varsa modele gönder
                    test_data = (np.array(ecg_data[-window_size:]) - np.mean(ecg_data[-window_size:])) / np.std(ecg_data[-window_size:])
                    test_data = test_data.reshape(1, window_size, 1)  # Modelin beklediği şekle dönüştür

                    prediction = model.predict(test_data)
                    predicted_class = class_labels[np.argmax(prediction, axis=1)[0]]
                    print("Predicted class:", predicted_class)

                    # Tahmin edilen sınıfı grafiğe ekle
                    root.after(0, text_pred.set_text, f'Predicted class: {predicted_class}')

            elif data == b'\xFA':  # Nabız değeri işareti
                pulse_data = ser.read(1)
                pulse_value = ord(pulse_data)
                print("Pulse value:", pulse_value)
                # Nabız değerini grafiğe ekle
                root.after(0, text_pulse.set_text, f'Pulse value: {pulse_value}')

            elif data == b'\xFB':  # Bilgi baytı işareti
                info_data = ser.read(1)
                if info_data == b'\x11':
                    print("Lead off detected")
                else:
                    print("Unknown info byte:", ord(info_data))
            else:
                print("Unknown data byte:", ord(data))
        time.sleep(0.01)  # CPU kullanımını azaltmak için kısa bir gecikme

def update_plot(ecg_segment, fs, lowcut, highcut):
    if filter_var.get():
        data_to_plot = butter_bandpass_filter(np.array(ecg_segment), lowcut, highcut, fs, order=4)
    else:
        data_to_plot = np.array(ecg_segment)

    r_peaks, q_peaks, s_peaks, pr_intervals, qrs_durations, qt_intervals = analyze_ecg(data_to_plot, fs)

    line.set_ydata(data_to_plot)
    line.set_xdata(range(len(data_to_plot)))
    ax.set_xlim(len(data_to_plot) - window_size, len(data_to_plot))
    ax.set_ylim(min(data_to_plot), max(data_to_plot))

    # R, Q, S tepe noktalarını işaretleme
    ax.plot(r_peaks, data_to_plot[r_peaks], 'ro', label='R Peaks')
    ax.plot(q_peaks, data_to_plot[q_peaks], 'go', label='Q Peaks')
    ax.plot(s_peaks, data_to_plot[s_peaks], 'bo', label='S Peaks')

    fig.canvas.draw()
    fig.canvas.flush_events()

def save_data():
    filename = filedialog.asksaveasfilename(defaultextension=".ecg", filetypes=[("ECG Files", "*.ecg")])
    if filename:
        ecg_json = json.dumps(ecg_data)
        encrypted_data = aes_encrypt(ecg_json.encode('utf-8'))
        with open(filename, 'wb') as file:
            file.write(encrypted_data)
        print(f"Data saved to {filename}")

def load_data():
    global ecg_data
    filename = filedialog.askopenfilename(filetypes=[("ECG Files", "*.ecg")])
    if filename:
        with open(filename, 'rb') as file:
            encrypted_data = file.read()
        try:
            decrypted_data = aes_decrypt(encrypted_data)
            ecg_data = json.loads(decrypted_data.decode('utf-8'))
            root.after(0, update_plot, ecg_data[-window_size:], 500, 0.5, 50.0)
            print(f"Data loaded from {filename}")
        except (cryptography.fernet.InvalidToken, json.JSONDecodeError) as e:
            print(f"Failed to load data: {e}")

def start_reading():
    threading.Thread(target=read_ecg_data, daemon=True).start()

# Tkinter arayüzü oluşturma
root = tk.Tk()
root.title("Gerçek Zamanlı Seri Port Verisi")

# Checkbutton ekleme
filter_var = tk.BooleanVar()
filter_checkbox = ttk.Checkbutton(root, text="Filtre Uygula", variable=filter_var)
filter_checkbox.pack()

# Başlat butonu ekleme
start_button = ttk.Button(root, text="Başlat", command=start_reading)
start_button.pack()

# Kaydet butonu ekleme
save_button = ttk.Button(root, text="Kaydet", command=save_data)
save_button.pack()

# Yükle butonu ekleme
load_button = ttk.Button(root, text="Yükle", command=load_data)
load_button.pack()

# Matplotlib grafiğini Tkinter penceresine ekleme
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Matplotlib araç çubuğunu ekleme
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Tkinter ana döngüsü
root.mainloop()

# Seri portu kapatma
ser.close()
