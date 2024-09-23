# patient.py

import mysql.connector
import serial
import tkinter as tk
from tkinter import ttk, simpledialog
import threading
import tenseal as ts
import time
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Seri port ve veri değişkenleri
ser = serial.Serial('COM5', 9600, timeout=1)
ecg_data = []
timestamps = []
current_id = None  # Şu anda kullanılan ID
is_reading = False
is_syncing = False

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
    cursor.execute("DELETE FROM encrypted_ecg_data WHERE id = %s", (record_id,))
    for vec in encrypted_data:
        encrypted_bytes = vec.serialize()
        cursor.execute("INSERT INTO encrypted_ecg_data (id, encrypted_value, timestamp) VALUES (%s, %s, %s)", (record_id, encrypted_bytes, timestamp))
    connection.commit()
    connection.close()
    print("Encrypted data saved to database")

# Context ve secret_key'i dosyalara kaydetme
def save_context_and_key():
    global context, secret_key
    with open("context.tenseal", "wb") as f:
        f.write(context.serialize())
    with open("secret_key.tenseal", "wb") as f:
        f.write(secret_key.serialize())

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
                if len(ecg_data) > 3000:  # Sadece son 3000 veriyi tutar
                    ecg_data[:] = ecg_data[-3000:]
                print("Wave data:", list(wave_data))
        time.sleep(0.01)  # CPU kullanımını azaltmak için kısa bir gecikme

    ser.close()  # Seri portu kapat

def start_reading():
    global is_reading, current_id
    current_id = simpledialog.askinteger("Input", "Enter the Patient ID:")
    if current_id is None:
        return
    is_reading = True
    threading.Thread(target=read_ecg_data).start()
    threading.Thread(target=plot_data_real_time).start()
    print("Started reading")

def stop_reading():
    global is_reading
    is_reading = False
    print("Stopped reading")

def encrypt_data(data):
    global encrypted_data, context, secret_key
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    secret_key = context.secret_key()  # Gizli anahtarı sakla
    public_context = ts.context_from(context.serialize(save_secret_key=False))
    encrypted_data = [ts.ckks_vector(public_context, [val]) for val in data]
    print("Data encrypted")
    return context

def sync_data():
    global context, is_syncing
    while is_syncing:
        if ecg_data:
            context = encrypt_data(ecg_data)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_encrypted_data(encrypted_data, timestamp, current_id)
            save_context_and_key()  # Context ve secret_key'i kaydet
        time.sleep(30)  # 30 saniyede bir senkronizasyon

def start_sync():
    global is_syncing
    is_syncing = True
    threading.Thread(target=sync_data).start()
    print("Started syncing")

def stop_sync():
    global is_syncing
    is_syncing = False
    print("Stopped syncing")

def plot_data_real_time():
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(ecg_data)
    ax.set_title(f"Patient ID: {current_id}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    while is_reading:
        line.set_ydata(ecg_data)
        line.set_xdata(range(len(ecg_data)))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1)
    plt.ioff()
    plt.show()

def on_closing():
    global is_reading, is_syncing
    is_reading = False
    is_syncing = False
    ser.close()
    root.destroy()

# Tkinter arayüzü oluşturma
root = tk.Tk()
root.title("Patient: Seri Port Verisi Okuma ve Şifreleme")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Buton çerçevesi
button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

# Butonlar
start_button = ttk.Button(button_frame, text="Start", command=start_reading)
start_button.grid(row=0, column=0, padx=5, pady=5)

sync_button = ttk.Button(button_frame, text="FHE Sync", command=start_sync)
sync_button.grid(row=0, column=1, padx=5, pady=5)

exit_button = ttk.Button(button_frame, text="Çıkış", command=on_closing)
exit_button.grid(row=0, column=2, padx=5, pady=5)

# Tkinter ana döngüsü
root.mainloop()
