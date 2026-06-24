<div align="center">

# HumanDetect

<p>
  <strong>HumanDetection AI</strong><br>
  Deteksi manusia realtime, tracking otomatis, hitung jumlah orang, status gerak, dan dashboard monitoring modern.
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Human%20Detection-111827?style=for-the-badge&logo=github&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Termux-0F172A?style=for-the-badge&logo=android&logoColor=white" alt="Platform">
  <img src="https://img.shields.io/badge/Mode-CLI%20%2B%20Live%20Camera-7C3AED?style=for-the-badge&logo=windowsterminal&logoColor=white" alt="CLI">
</p>

</div>

---

## Tentang Project

`HumanDetect` adalah aplikasi Python untuk mendeteksi manusia secara realtime menggunakan kamera.  
Aplikasi ini dirancang agar:

- ringan dan mudah dijalankan
- punya tampilan live monitoring yang rapi
- menghitung jumlah orang aktif dan total orang unik otomatis
- mendeteksi status pergerakan manusia
- bisa dijalankan di PC dan juga mode headless seperti Termux

Project ini cocok untuk:

- monitoring ruangan
- prototipe CCTV AI
- analisis okupansi
- sistem hitung orang realtime
- eksperimen computer vision berbasis YOLOv8

---

## Fitur Utama

- Deteksi manusia realtime dari kamera
- Tracking otomatis per orang dengan `ID`
- Hitung `orang aktif`, `total terdeteksi`, dan `puncak okupansi`
- Status gerak per orang: `Dipantau`, `Aktif`, `Bergerak`
- Dashboard modern langsung di tampilan kamera
- Info `FPS` dan `uptime` sesi
- Bisa simpan hasil video anotasi ke file `.mp4`
- Bisa dijalankan dengan backend:
  - `ultralytics`
  - `onnx`
  - `tflite`
- Support mode singkat:
  - `python human.py`

---

## Struktur File

```text
HumanDetect/
├─ human.py
├─ human_detect_cli.py
├─ yolov8n.pt
└─ README.md
```

- `human.py` = launcher singkat
- `human_detect_cli.py` = aplikasi utama
- `yolov8n.pt` = model YOLOv8n default

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/floryid/HumanDetect.git
cd HumanDetect
```

### 2. Install Dependensi

Untuk penggunaan PC dengan backend `ultralytics`:

```bash
pip install numpy pillow opencv-python ultralytics lap
```

Jika ingin backend `onnx`:

```bash
pip install numpy pillow opencv-python onnxruntime
```

Jika ingin backend `tflite`:

```bash
pip install numpy pillow opencv-python tflite-runtime
```

### 3. Jalankan Aplikasi

Perintah paling singkat:

```bash
python human.py
```

Atau langsung file utama:

```bash
python human_detect_cli.py
```

---

## Cara Penggunaan

### Jalankan Normal

```bash
python human.py
```

### Pilih Kamera Tertentu

```bash
python human.py --camera 0
```

Jika kamera default tidak terbaca, coba:

```bash
python human.py --camera 1
```

### Jalankan Tanpa Tampilan GUI

Mode ini cocok untuk server atau Termux:

```bash
python human.py --no-gui
```

### Simpan Video Hasil Deteksi

```bash
python human.py --save output.mp4
```

### Gunakan Backend Ultralytics

```bash
python human.py --backend ultralytics --model yolov8n.pt
```

### Gunakan Backend ONNX

```bash
python human.py --backend onnx --model yolov8n.onnx
```

### Gunakan Backend TFLite

```bash
python human.py --backend tflite --model yolov8n.tflite
```

---

## Opsi CLI Penting

Berikut opsi yang paling sering dipakai:

| Opsi | Fungsi |
|---|---|
| `--backend` | Memilih backend: `auto`, `ultralytics`, `onnx`, `tflite` |
| `--source` | Sumber frame: `opencv` atau `termux-photo` |
| `--model` | Path model YOLO |
| `--camera` | Index kamera |
| `--conf` | Confidence threshold |
| `--iou` | IoU threshold |
| `--imgsz` | Ukuran input model |
| `--no-gui` | Jalankan tanpa jendela kamera |
| `--print-every` | Interval log ke terminal |
| `--movement-px` | Ambang pixel untuk status bergerak |
| `--movement-cooldown` | Jeda minimum event gerak |
| `--save` | Simpan video anotasi |

Untuk melihat semua opsi:

```bash
python human.py --help
```

---

## Tampilan Monitoring

Saat aplikasi berjalan, tampilan live akan menunjukkan:

- jumlah orang yang sedang terlihat
- total orang unik terdeteksi selama sesi
- puncak jumlah orang
- FPS realtime
- uptime monitoring
- daftar orang aktif
- status tiap orang

Contoh status:

- `Dipantau` = terdeteksi stabil
- `Aktif` = pernah bergerak dalam sesi
- `Bergerak` = sedang terdeteksi berpindah posisi

---

## Output Terminal

Contoh log terminal:

```text
terlihat_sekarang=2 orang | total_terdeteksi=6 orang | puncak=3 orang | fps=24.8 | uptime=01:23 | orang_baru=+1
 - Orang #1 | akurasi=0.91 | status=Dipantau | pantau=00:17
 - Orang #2 | akurasi=0.88 | status=Bergerak | pantau=00:09
```

Log ini berguna untuk:

- monitoring tanpa GUI
- debugging
- integrasi ke sistem lain

---

## Menjalankan di Termux

Project ini juga mendukung mode Termux dengan sumber kamera `termux-camera-photo`.

### Install dasar di Termux

```bash
pkg update
pkg install python termux-api
pip install numpy pillow onnxruntime
```

### Jalankan mode Termux

```bash
python human.py --source termux-photo --backend onnx --model yolov8n.onnx --no-gui
```

Catatan:

- mode Termux lebih cocok pakai `onnx` atau `tflite`
- backend `ultralytics` biasanya lebih nyaman dipakai di PC/laptop

---

## Rekomendasi Penggunaan

Untuk hasil terbaik:

- gunakan ruangan dengan pencahayaan cukup
- posisikan kamera stabil
- gunakan model yang sesuai perangkat
- untuk PC, backend `ultralytics` adalah pilihan termudah
- untuk perangkat ringan, gunakan `onnx` atau `tflite`

---

## Troubleshooting

### Kamera tidak terbuka

Coba ganti index kamera:

```bash
python human.py --camera 1
```

### Modul belum terinstall

Install ulang dependensi:

```bash
pip install numpy pillow opencv-python ultralytics lap
```

### Ingin lihat semua opsi

```bash
python human.py --help
```

### Ingin jalankan paling cepat

```bash
python human.py
```

---

## Command Singkat

Project ini sudah menyediakan launcher singkat:

```bash
python human.py
```

Jadi kamu tidak perlu lagi mengetik:

```bash
python human_detect_cli.py
```

---

## Tech Stack

- Python
- YOLOv8
- OpenCV
- NumPy
- ONNX Runtime
- TFLite Runtime

---

## Repository

- GitHub: [floryid/HumanDetect](https://github.com/floryid/HumanDetect)

---

## Lisensi

Project ini menggunakan lisensi sesuai pengaturan repository GitHub kamu.
