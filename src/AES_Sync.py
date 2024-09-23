# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:52:07 2024

@author: BBYCLUB_2022
"""

import mysql.connector
import serial
import tkinter as tk
from tkinter import ttk, simpledialog
import queue
import threading
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
from cryptography.fernet import Fernet

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
is_syncing = False
is_plotting = False

# Fernet anahtarı oluşturma
key = Fernet.generate_key()
cipher = Fernet(key)

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
        cursor.execute("INSERT INTO encrypted_ecg_data (id, encrypted_value, timestamp) VALUES (%s, %s, %s)", (record_id, vec, timestamp))
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
        encrypted_data = [enc_data[0] for enc_data in result]
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
    global ecg_data, timestamps, is_syncing

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
                if is_syncing:
                    encrypted_data = encrypt_data(list(wave_data))
                    save_encrypted_data(encrypted_data, timestamp, current_id)
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
    global is_reading, current_id
    current_id = simpledialog.askinteger("Input", "Enter the ID:")
    if current_id is None:
        return
    is_reading = True
    threading.Thread(target=read_ecg_data).start()
    print("Started reading")

def start_syncing():
    global is_syncing
    is_syncing = True
    print("Started syncing")

def start_plotting():
    global is_plotting
    is_plotting = True
    ani.event_source.start()
    print("Started plotting")

def stop_all():
    global is_reading, is_syncing, is_plotting
    is_reading = False
    is_syncing = False
    is_plotting = False
    ani.event_source.stop()
    print("Stopped all operations")

def encrypt_data(data):
    encrypted_data = [cipher.encrypt(bytes([val])) for val in data]
    print("Data encrypted")
    return encrypted_data

def decrypt_data(encrypted_data):
    decrypted_data = [cipher.decrypt(enc_data)[0] for enc_data in encrypted_data]
    print("Data decrypted")
    return decrypted_data

def update_plot(frame):
    if is_plotting:
        if len(ecg_data) > 0:
            original_data = ecg_data[-window_size:]
            record_id = current_id
            encrypted_data = load_encrypted_data(record_id)
            decrypted_data = decrypt_data(encrypted_data) if encrypted_data else []
            line1.set_data(range(len(original_data)), original_data)
            line2.set_data(range(len(decrypted_data)), decrypted_data)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
    return line1, line2

def on_closing():
    global is_reading
    is_reading = False
    ser.close()
    root.destroy()

# Tkinter arayüzü oluşturma
root = tk.Tk()
root.title("Monitorıing Real-Time ECG")

root.protocol("WM_DELETE_WINDOW", on_closing)

# Buton çerçevesi
button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

# Butonlar
start_button = ttk.Button(button_frame, text="Start", command=start_reading)
start_button.grid(row=0, column=0, padx=5, pady=5)

sync_button = ttk.Button(button_frame, text="Sync", command=start_syncing)
sync_button.grid(row=0, column=1, padx=5, pady=5)

plot_button = ttk.Button(button_frame, text="Plot", command=start_plotting)
plot_button.grid(row=0, column=2, padx=5, pady=5)

stop_button = ttk.Button(button_frame, text="Stop", command=stop_all)
stop_button.grid(row=0, column=3, padx=5, pady=5)

# Accuracy label
accuracy_label = ttk.Label(root, text="Accuracy: N/A")
accuracy_label.pack(side=tk.BOTTOM)

# Matplotlib gerçek zamanlı grafik
fig, (ax1, ax2) = plt.subplots(2, 1)
line1, = ax1.plot([], [], label='Original Data')
line2, = ax2.plot([], [], label='Decrypted Data')
ax1.set_ylim(-10, 300)
ax2.set_ylim(-10, 300)

ani = FuncAnimation(fig, update_plot, blit=True, interval=1000)
ani.event_source.stop()

# Tkinter ana döngüsü
plt.show(block=False)
root.mainloop()
