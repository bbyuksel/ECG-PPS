# doctor.py

import mysql.connector
import tkinter as tk
from tkinter import ttk, simpledialog
import tenseal as ts
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Context ve secret_key'i global olarak tanımlayın
context = None
secret_key = None

# MySQL bağlantısı ve tablo oluşturma
def create_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecg_data"
    )
    return connection

def load_context_and_key():
    global context, secret_key
    # context ve secret_key dosyalardan yükleyin veya başka bir uygun yolla oluşturun
    # Örnek olarak, dosyadan yükleme:
    with open("context.tenseal", "rb") as f:
        context_data = f.read()
        context = ts.context_from(context_data)
    with open("secret_key.tenseal", "rb") as f:
        secret_key_data = f.read()
        context.load(secret_key_data)

# Veriyi okuma ve çözme
def load_encrypted_data(record_id):
    connection = create_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT encrypted_value, timestamp FROM encrypted_ecg_data WHERE id = %s", (record_id,))
    result = cursor.fetchall()
    connection.close()
    if result:
        encrypted_data = [ts.ckks_vector_from(context, bytes(enc_data[0])) for enc_data in result]
        timestamps = [enc_data[1] for enc_data in result]
        return encrypted_data, timestamps
    else:
        print("No data found with the given ID")
        return [], []

def decrypt_data(context, encrypted_data):
    decrypted_data = [vec.decrypt(secret_key=context.secret_key()) for vec in encrypted_data]
    decrypted_data = [item[0] for item in decrypted_data]
    print("Data decrypted")
    return decrypted_data

def plot_data_comparison(original_data, decrypted_data, timestamps, patient_id):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(original_data)
    axs[0].set_title(f"Original Data for Patient ID: {patient_id}")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Amplitude")
    axs[1].plot(decrypted_data)
    axs[1].set_title(f"Decrypted Data for Patient ID: {patient_id}")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("Amplitude")
    for ax in axs:
        for i, ts in enumerate(timestamps):
            ax.annotate(ts.strftime('%H:%M:%S'), (i, decrypted_data[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.show()

def start_fhe_decrypt():
    global context, secret_key
    load_context_and_key()  # Context ve secret_key'i yükleyin
    patient_id = simpledialog.askinteger("Input", "Enter the Patient ID:")
    if patient_id is not None:
        encrypted_data, timestamps = load_encrypted_data(patient_id)  # Veriyi yükle
        if encrypted_data:
            decrypted_data = decrypt_data(context, encrypted_data)
            plot_data_comparison([], decrypted_data, timestamps, patient_id)

def on_closing():
    root.destroy()

# Tkinter arayüzü oluşturma
root = tk.Tk()
root.title("Doctor: Veritabanından Veri Okuma ve Çözme")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Buton çerçevesi
button_frame = ttk.Frame(root)
button_frame.pack(side=tk.BOTTOM)

# Butonlar
decrypt_button = ttk.Button(button_frame, text="Load and Decrypt Data", command=start_fhe_decrypt)
decrypt_button.grid(row=0, column=0, padx=5, pady=5)

exit_button = ttk.Button(button_frame, text="Çıkış", command=on_closing)
exit_button.grid(row=0, column=1, padx=5, pady=5)

# Tkinter ana döngüsü
root.mainloop()
