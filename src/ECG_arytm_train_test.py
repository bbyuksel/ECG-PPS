# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:38:47 2024

@author: BBYCLUB_2022
"""

import numpy as np
import wfdb
from wfdb import processing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# MIT-BIH Arrhythmia Database'ten verileri indir
record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/100')
annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/100', 'atr')

# EKG sinyalini ve etiketleri al
ecg_signal = record.p_signal[:, 0]
labels = annotation.symbol

# R-dalgası tespiti
r_peaks = processing.gqrs_detect(sig=ecg_signal, fs=record.fs)

# R-dalgalarının etrafında veri toplama
segments = []
segment_labels = []

window_size = 180  # Örnek sayısı (2 saniyelik pencere)

for peak in r_peaks:
    if peak > window_size // 2 and peak < len(ecg_signal) - window_size // 2:
        segment = ecg_signal[peak - window_size // 2: peak + window_size // 2]
        segments.append(segment)
        segment_labels.append(labels[0])  # Etiketleri uygun şekilde eşleştirin

segments = np.array(segments)
segment_labels = np.array(segment_labels)

# Sadece belirli etiketleri içeren verileri kullanma
label_mapping = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}
filtered_segments = []
filtered_labels = []

for segment, label in zip(segments, segment_labels):
    if label in label_mapping:
        filtered_segments.append(segment)
        filtered_labels.append(label_mapping[label])

filtered_segments = np.array(filtered_segments)
filtered_labels = np.array(filtered_labels)

# Verileri normalize et
filtered_segments = (filtered_segments - np.mean(filtered_segments)) / np.std(filtered_segments)

# Etiketleri one-hot encode et
filtered_labels = to_categorical(filtered_labels)

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(filtered_segments, filtered_labels, test_size=0.2, random_state=42)

# Veriyi uygun şekle dönüştür
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#%%
# Basit bir CNN modeli tanımla
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Modeli kaydet
model.save('model.h5')

#%%
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:12:45 2024

@author: BBYCLUB_2022
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Seri port ayarları
port = 'COM5'  # Seri portunuzu buraya yazın
baud_rate = 9600

# Seri portu aç
ser = serial.Serial(port, baud_rate, timeout=1)

# EKG verilerini saklamak için liste
ecg_data = []

# Modeli yükleyin
model = load_model('model.h5')

# Matplotlib ayarları
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_ylim(-128, 128)
ax.set_xlim(0, 300)

# Modülü başlatma ve veri okuma fonksiyonu
def read_ecg_data():
    global ecg_data

    while True:
        if ser.in_waiting > 0:
            data = ser.read(1)
            if data == b'\xF8':  # Dalga örnekleme noktası işareti
                wave_data = ser.read(50)  # 50 Hz hızında 50 örnek okur
                ecg_data.extend(list(wave_data))
                if len(ecg_data) > 3000:  # Sadece son 3000 veriyi tutar
                    ecg_data[:] = ecg_data[-3000:]
                # Ekranı güncelle
                line.set_ydata(ecg_data[-300:])
                line.set_xdata(range(len(ecg_data[-300:])))
                ax.set_xlim(len(ecg_data[-300:])-300, len(ecg_data[-300:]))
                ax.set_ylim(min(ecg_data[-300:]), max(ecg_data[-300:]))
                fig.canvas.draw()
                fig.canvas.flush_events()
                print("Wave data:", list(wave_data))

                # Veriyi normalize edin ve modele gönderin
                if len(ecg_data) >= 300:  # Yeterli veri varsa modele gönder
                    test_data = np.array(ecg_data[-300:])
                    test_data = (test_data - np.mean(test_data)) / np.std(test_data)
                    test_data = test_data.reshape(1, -1, 1)  # Modelin beklediği şekle dönüştür

                    prediction = model.predict(test_data)
                    predicted_class = np.argmax(prediction, axis=1)
                    print("Predicted class:", predicted_class)

            elif data == b'\xFA':  # Nabız değeri işareti
                pulse_data = ser.read(1)
                print("Pulse value:", ord(pulse_data))
            elif data == b'\xFB':  # Bilgi baytı işareti
                info_data = ser.read(1)
                if info_data == b'\x11':
                    print("Lead off detected")
                else:
                    print("Unknown info byte:", ord(info_data))
            else:
                print("Unknown data byte:", ord(data))
        time.sleep(0.01)  # CPU kullanımını azaltmak için kısa bir gecikme

try:
    read_ecg_data()
except KeyboardInterrupt:
    print("Program durduruldu.")
finally:
    ser.close()
