# 🌿 E-Dravya

**E-Dravya** is an AI + IoT powered system to detect **adulteration in medicinal herbs** using sensor data and cloud-based machine learning.  

Built for SIH 2025, it combines **ESP32 firmware**, **cloud inference**, and a **Next.js dashboard** to make quality control simple, fast, and reliable.

## 🚀 Features

- 📡 **ESP32 Logger** - reads pH, TDS, ORP, temperature, and RGB values from sensors and sends them to the cloud.
- ☁️ **Cloud Inference** - AWS Lambda runs trained ML models to classify herbs and detect adulteration.
- 🤖 **ML Training** - Random Forest model trained on curated datasets with preprocessing + feature engineering.
- 📊 **Dashboard** - modern Next.js + shadcn/ui interface for live scans, model monitoring, and retraining workflows.
- 🗃 **Data Pipeline** - DynamoDB for scan history, with tools to export clean CSVs for retraining.

## 🛠 Tech Stack

- **Hardware**: ESP32, sensors (pH, TDS, ORP, RGB, Temperature)  
- **Cloud**: AWS Lambda + API Gateway + DynamoDB + S3  
- **ML**: scikit-learn (RandomForest), preprocessing pipeline  
- **Frontend**: Next.js, TailwindCSS, shadcn/ui, Recharts  
- **Utilities**: boto3, pandas, joblib  

## 📊 Dashboard Preview

A Next.js + shadcn/ui dashboard with:
- KPI cards (total scans, % adulterated, avg confidence)
- History tables with filters & CSV export
- Device pages
- Model monitoring (confusion matrix, feature importance)

## 🌟 Vision

Bringing trust back to traditional medicine by fusing IoT, AI, and clean design, E-Dravya is all about making quality products **transparent and accessible.**

## 📜 License

MIT License © 2025 Judas


