# Real-Time Object Detection with Webcam

> **TL;DR** — Real-time object detection menggunakan YOLOv3-tiny dan OpenCV untuk mendeteksi objek dari webcam dengan bounding box dan label.

---

## 🔬 Business Question

Bagaimana mendeteksi dan mengenali objek secara real-time dari input webcam untuk keperluan monitoring atau sistem cerdas?

**Tujuan:**
- Implementasi deteksi objek real-time menggunakan YOLO
- Menggunakan webcam sebagai input sumber
- Menampilkan hasil deteksi dengan bounding box dan label

---

## 🧮 Metodologi

```
1. Load YOLOv3-tiny model (pre-trained on COCO dataset)
2. Capture frame dari webcam
3. Pre-processing: resize, normalize, create blob
4. Forward pass melalui YOLO network
5. Post-processing: filter confidence, apply NMS, draw boxes
6. Display hasil real-time
```

### Model Details

| Aspek | Detail |
|---|---|
| **Model** | YOLOv3-tiny |
| **Dataset** | COCO (80 classes) |
| **Input** | Webcam (default camera 0) |
| **Backend** | OpenCV DNN |

---

## 💡 Key Findings

1. YOLOv3-tiny cukup cepat untuk real-time di CPU
2. OpenCV DNN backend memungkinkan inferensi tanpa dependencies berat
3. Model bisa detect 80 objek (person, car, dog, dll)

---

## 🚀 Cara Menjalankan

```bash
# Clone repo
git clone https://github.com/henryferdianto/Deteksi-Objek-Real-Time-dengan-Webcam.git
cd Deteksi-Objek-Real-Time-dengan-Webcam

# Install dependencies
pip install -r requirements.txt

# Jalankan (download model dulu via notebook atau manual)
python detect.py --weights yolo/yolov3-tiny.weights --cfg yolo/yolov3-tiny.cfg --names yolo/coco.names
```

Atau jalankan `main.ipynb` untuk download model otomatis.

---

## ⚠️ Limitations

- Butuh download YOLO weights (~34MB) terpisah
- Performa di CPU terbatas
- Akurasi lebih rendah dari YOLOv3 full

---

## 🛠️ Tech Stack

- Python
- OpenCV (cv2)
- NumPy
- YOLOv3-tiny (pre-trained)

---

## 📜 License

MIT License — lihat [LICENSE](LICENSE).
---

## 🎯 Demo & Results

### Detection Sample

![Detection Result](reports/figures/detection-sample.png)

*Contoh hasil deteksi objek real-time menggunakan YOLOv3-tiny*

### Performance Notes

| Aspect | Detail |
|---|---|
| FPS (approx) | TBD |
| mAP | TBD |

> Screenshot dari hasil running `detect.py` bisa ditambahkan ke `reports/figures/`
