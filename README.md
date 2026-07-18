# 🌶️ Klasifikasi Penyakit Daun Cabai Menggunakan Transfer Learning

Proyek ini merupakan implementasi **Deep Learning** untuk mengklasifikasikan penyakit daun cabai menggunakan metode **Transfer Learning** dengan membandingkan dua model, yaitu **MobileNetV2** dan **EfficientNetB0**.

Model dikembangkan menggunakan **TensorFlow** dan **Keras** pada lingkungan **Google Colab**.

---

## 📌 Deskripsi Proyek

Penyakit pada daun cabai dapat menurunkan kualitas serta hasil panen. Oleh karena itu, diperlukan sistem yang mampu membantu proses identifikasi penyakit secara otomatis berdasarkan citra daun.

Pada proyek ini dikembangkan model klasifikasi citra daun cabai ke dalam tiga kelas:

- 🌿 Healthy
- 🍃 Leaf Curl
- 🍂 Leaf Spot

Dua model Transfer Learning dibandingkan untuk mengetahui model yang memiliki performa terbaik.

---

# 📂 Dataset

Dataset berasal dari Kaggle.

**Link Dataset**

https://www.kaggle.com/datasets/suraj520/chili-plant-disease-dataset

Struktur dataset yang digunakan:

```text
Dataset/
│
├── train/
│   ├── healthy/
│   ├── leaf curl/
│   └── leaf spot/
│
├── val/
│   ├── healthy/
│   ├── leaf curl/
│   └── leaf spot/
│
└── test/
    ├── healthy/
    ├── leaf curl/
    └── leaf spot/
```

---

# 🧠 Model yang Digunakan

Penelitian menggunakan dua model Transfer Learning.

| Model | Status |
|--------|--------|
| MobileNetV2 | ✅ |
| EfficientNetB0 | ✅ |

---

# ⚙️ Library

Library yang digunakan pada proyek ini antara lain:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- OpenCV

---

# 📊 Workflow Proyek

```text
Business Understanding
        │
        ▼
Data Understanding
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Data Preparation
        │
        ▼
Modeling
   ├── MobileNetV2
   └── EfficientNetB0
        │
        ▼
Evaluation
        │
        ▼
Deployment
```

---

# 📈 Tahapan Penelitian

## 1. Business Understanding

Menentukan permasalahan, tujuan penelitian, serta solusi yang akan dikembangkan menggunakan Artificial Intelligence.

---

## 2. Data Understanding

Melakukan analisis terhadap dataset yang digunakan.

- Struktur Dataset
- Jumlah Kelas
- Jumlah Data
- Target Klasifikasi

---

## 3. Exploratory Data Analysis

Melakukan eksplorasi dataset.

- Distribusi Data
- Visualisasi Dataset
- Analisis Ketidakseimbangan Data

---

## 4. Data Preparation

Tahapan preprocessing meliputi:

- Resize gambar menjadi **224×224**
- Normalisasi piksel
- Data Augmentation
- Load Dataset

---

## 5. Modeling

Dua model dibandingkan.

### MobileNetV2

- Pre-trained ImageNet
- Global Average Pooling
- Dense Layer
- Softmax

### EfficientNetB0

- Pre-trained ImageNet
- Global Average Pooling
- Dense Layer
- Softmax

---

## 6. Evaluation

Model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 7. Deployment

Model terbaik digunakan untuk memprediksi gambar daun cabai yang diunggah pengguna melalui Google Colab.

---

# 📁 Struktur Repository

```text
.
├── Dataset/
├── images/
│   ├── distribusi_data.png
│   ├── sample_dataset.png
│   ├── data_augmentation.png
│   ├── mobilenet_accuracy.png
│   ├── mobilenet_loss.png
│   ├── efficientnet_accuracy.png
│   ├── efficientnet_loss.png
│   ├── accuracy_comparison.png
│   ├── loss_comparison.png
│   ├── confusion_matrix_mobilenet.png
│   ├── confusion_matrix_efficientnet.png
│   └── hasil_prediksi.png
│
├── notebook.ipynb
├── best_model.keras
├── hasil_perbandingan_model.csv
├── Laporan_uas.md
├── README.md
└── requirements.txt
```

---

# 📷 Dokumentasi

## Contoh Dataset

<p align="center">
<img src="images/sample_dataset.png" width="700">
</p>

---

## Grafik Accuracy

<p align="center">
<img src="images/accuracy_comparison.png" width="700">
</p>

---

## Grafik Loss

<p align="center">
<img src="images/loss_comparison.png" width="700">
</p>

---

## Confusion Matrix

<p align="center">
<img src="images/confusion_matrix_mobilenet.png" width="450">
<img src="images/confusion_matrix_efficientnet.png" width="450">
</p>

---

## Contoh Hasil Prediksi

<p align="center">
<img src="images/hasil_prediksi.png" width="450">
</p>

---

# 📌 Hasil

Model berhasil mengklasifikasikan citra daun cabai menjadi tiga kelas:

- Healthy
- Leaf Curl
- Leaf Spot

Selanjutnya dilakukan perbandingan performa antara MobileNetV2 dan EfficientNetB0 menggunakan metrik evaluasi sehingga diperoleh model terbaik untuk proses deployment.

---

# 👨‍💻 Author

**Rio Cahya Ramadhan**

Universitas Muhammadiyah Sukabumi

Program Studi Informatika

---

# 📚 Referensi

- Howard, A., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.*
- Ferentinos, K. P. (2018). *Deep Learning Models for Plant Disease Detection and Diagnosis.*
- Shoaib, M., Shah, B., El-Sappagh, S., Ali, A., Ullah, A., Alenezi, F., Gechev, T., Hussain, T., & Ali, F. (2023). An Advanced Deep Learning Models-Based Plant Disease Detection: A Review of Recent Research. Frontiers in Plant Science, 14, 1158933.
---

⭐ **Jika repository ini bermanfaat, jangan lupa berikan star pada repository GitHub ini.**