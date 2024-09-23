# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:24:54 2024

@author: BBYCLUB_2022
"""

import mysql.connector
import asyncio
import serial
import tkinter as tk
from tkinter import ttk, simpledialog
import queue
import threading
import tenseal as ts
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Seri port ve veri değişkenleri
ser = serial.Serial('COM5', 9600, timeout=1)
ecg_data = []
timestamps = []
window_size = 3000
current_id = None  # Şu anda kullanılan ID

# Veri kuyruğu oluşturma
data_queue = queue.Queue()
stored_data = []
encrypted_data = []
decrypted_data = []
is_reading = False

context = None  # Context nesnesi global olarak tanımlanır
secret_key = None  # Gizli anahtar global olarak tanımlanır

# MySQL bağlantısı ve tablo oluşturma
def create_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecg_data"
    )
    return connection

def create_table():
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS encrypted_ecg_data (
        id INT,
        encrypted_value LONGBLOB,
        timestamp DATETIME
    )
    """)
    connection.close()

create_table()

# Veriyi saklama
def save_encrypted_data(encrypted_data, timestamp, record_id):
    connection = create_db_connection()
    cursor = connection.cursor()
    for vec in encrypted_data:
        encrypted_bytes = vec.serialize()
        cursor.execute("INSERT INTO encrypted_ecg_data (id, encrypted_value, timestamp) VALUES (%s, %s, %s)", (record_id, encrypted_bytes, timestamp))
    connection.commit()
    connection.close()
    print("Encrypted data saved to database")

# Veriyi okuma ve çözme
def load_encrypted_data(record_id):
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT encrypted_value FROM encrypted_ecg_data WHERE id = %s", (record_id,))
    result = cursor.fetchall()
    connection.close()
    if result:
        encrypted_data = [ts.ckks_vector_from(context, enc_data[0]) for enc_data in result]
        return encrypted_data
    else:
        print("No data found with the given ID")
        return []

# Bant geçiş filtresi
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Modülü başlatma ve veri okuma fonksiyonu
def read_ecg_data():
    global ecg_data, timestamps

    while is_reading:
        if ser.in_waiting > 0:
            data = ser.read(1)
            if data == b'\xF8':  # Dalga örnekleme noktası işareti
                wave_data = ser.read(50)  # 50 Hz hızında 50 örnek okur
                ecg_data.extend(list(wave_data))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                timestamps.extend([timestamp] * len(wave_data))
                stored_data.extend(list(wave_data))  # Verileri kaydet
                if len(ecg_data) > 3000:  # Sadece son 3000 veriyi tutar
                    ecg_data[:] = ecg_data[-3000:]
                print("Wave data:", list(wave_data))
            elif data == b'\xFA':  # Nabız değeri işareti
                pulse_data = ser.read(1)
                pulse_value = ord(pulse_data)
                print("Pulse value:", pulse_value)
            elif data == b'\xFB':  # Bilgi baytı işareti
                info_data = ser.read(1)
                if info_data == b'\x11':
                    print("Lead off detected")
                else:
                    print("Unknown info byte:", ord(info_data))
            else:
                print("Unknown data byte:", ord(data))
        time.sleep(0.01)  # CPU kullanımını azaltmak için kısa bir gecikme

    ser.close()  # Seri portu kapat

def start_reading():
    global is_reading
    is_reading = True
    threading.Thread(target=read_ecg_data).start()
    print("Started reading")

def stop_reading():
    global is_reading, current_id
    is_reading = False
    print("Stopped reading, data saved")
    context = encrypt_data(stored_data)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_encrypted_data(encrypted_data, timestamp, current_id)  # Şifrelenmiş veriyi sakla

def encrypt_data(data):
    global encrypted_data, context, secret_key
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    secret_key = context.secret_key()  # Gizli anahtarı sakla
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

def start_fhe_decrypt():
    global context
    record_id = simpledialog.askinteger("Input", "Enter the ID:")
    if record_id is not None:
        encrypted_data = load_encrypted_data(record_id)  # Veriyi yükle
        if encrypted_data:
            decrypted_data = decrypt_data(context)
            accuracy = calculate_accuracy(stored_data, decrypted_data)
            accuracy_label.config(text=f'Accuracy: {accuracy:.4f}')
            plot_data_comparison(stored_data, decrypted_data, "Original Data", "Decrypted Data")

def plot_data_comparison(original_data, decrypted_data, title1, title2):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(original_data)
    axs[0].set_title(title1)
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Amplitude")
    axs[1].plot(decrypted_data)
    axs[1].set_title(title2)
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def on_closing():
    global is_reading
    is_reading = False
    ser.close()
    root.destroy()

# Tkinter arayüzü oluşturma
root = tk.Tk()
root.title("Seri Port Verisi Okuma ve Şifreleme")

# Program başlatıldığında ID isteme
current_id = simpledialog.askinteger("Input", "Enter the ID:")
if current_id is None:
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Buton çerçevesi
button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

# Butonlar
start_button = ttk.Button(button_frame, text="Start", command=start_reading)
start_button.grid(row=0, column=0, padx=5, pady=5)

stop_button = ttk.Button(button_frame, text="Stop", command=stop_reading)
stop_button.grid(row=0, column=1, padx=5, pady=5)

decrypt_button = ttk.Button(button_frame, text="FHE Load", command=start_fhe_decrypt)
decrypt_button.grid(row=0, column=2, padx=5, pady=5)

exit_button = ttk.Button(button_frame, text="Çıkış", command=on_closing)
exit_button.grid(row=0, column=3, padx=5, pady=5)

# Accuracy label
accuracy_label = ttk.Label(root, text="Accuracy: N/A")
accuracy_label.pack(side=tk.BOTTOM)

# Tkinter ana döngüsü
root.mainloop()
