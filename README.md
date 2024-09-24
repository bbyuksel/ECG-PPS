# ECG-PPS: Real-Time ECG Monitoring and Analysis System

![Project Logo](assets/logo.png)
 <!-- Add your project logo here -->

 
## Introduction
This work is accepted at the SINCONF 2024

## Overview
This project implements a real-time ECG monitoring system using Fully Homomorphic Encryption (FHE) for privacy-preserving analysis. The system captures ECG signals, encrypts them, and stores them securely while enabling real-time visualization and disease detection.

## Features
- 🩺 **Real-Time ECG Monitoring:** Continuously captures and displays ECG signals.
- 🔐 **Privacy-Preserving Analysis:** Uses FHE to perform computations on encrypted data.
- 📊 **Disease Detection:** Employs a CNN model trained on the MIT-BIH Arrhythmia Database.
- 📈 **Data Visualization:** Real-time plotting of original and decrypted ECG data.

## Project Structure
```plaintext
ECG-PPS/
│
├── src/                  # Source code for the project
│   └── detailed_FHE_with_Figures_2.py
│
├── data/                 # Sample datasets and input data
│
├── docs/                 # Documentation and additional resources
│
└── README.md             # Project documentation
