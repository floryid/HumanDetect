<div align="center">

# HumanDetect

<p>
  <strong>HumanDetection AI</strong><br>
  Aplikasi deteksi manusia realtime dengan tracking otomatis, hitung jumlah orang, status gerak, dan dashboard monitoring modern.
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Human%20Detection-111827?style=for-the-badge&logo=github&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Platform-PC%20%7C%20Termux-0F172A?style=for-the-badge&logo=android&logoColor=white" alt="Platform">
  <img src="https://img.shields.io/badge/Mode-CLI%20%2B%20Live%20Camera-7C3AED?style=for-the-badge&logo=windowsterminal&logoColor=white" alt="CLI">
</p>

</div>

---

## Ringkasan

`HumanDetect` adalah aplikasi Python untuk mendeteksi manusia dari kamera secara realtime.  
Aplikasi ini dibuat agar mudah dijalankan, mudah dipahami, dan bisa dipakai baik di:

- PC / laptop
- server / headless mode
- Termux Android

Fungsi utama aplikasi:

- mendeteksi manusia dari kamera
- memberi `ID` unik per orang
- menghitung jumlah orang aktif
- menghitung total orang unik selama sesi
- menandai status gerak seperti `Dipantau`, `Aktif`, dan `Bergerak`
- menampilkan dashboard monitoring yang rapi

Project ini cocok untuk:

- monitoring ruangan
- eksperimen CCTV AI
- sistem hitung orang
- analisis okupansi
- pembelajaran computer vision berbasis YOLOv8

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Struktur Project](#struktur-project)
- [Cara Kerja Singkat](#cara-kerja-singkat)
- [Instalasi di PC](#instalasi-di-pc)
- [Instalasi di Termux Android](#instalasi-di-termux-android)
- [Cara Menjalankan Aplikasi](#cara-menjalankan-aplikasi)
- [Penjelasan Backend](#penjelasan-backend)
- [Opsi CLI Penting](#opsi-cli-penting)
- [Tampilan Monitoring](#tampilan-monitoring)
- [Contoh Output Terminal](#contoh-output-terminal)
- [Troubleshooting](#troubleshooting)

---

## Fitur Utama

- Deteksi manusia realtime dari kamera
- Tracking otomatis per orang dengan `ID`
- Hitung `orang aktif`, `total terdeteksi`, dan `puncak okupansi`
- Status pergerakan manusia:
  - `Dipantau`
  - `Aktif`
  - `Bergerak`
- Dashboard live yang modern dan rapi
- Informasi `FPS` dan `uptime`
- Bisa menyimpan hasil video anotasi ke file `.mp4`
- Bisa dijalankan dengan backend:
  - `ultralytics`
  - `onnx`
  - `tflite`
- Punya launcher singkat:
  - `python human.py`

---

## Struktur Project

```text
HumanDetect/
├─ human.py
├─ human_detect_cli.py
├─ yolov8n.pt
└─ README.md
```

Penjelasan file:

- `human.py`
  - launcher singkat
  - file ini yang paling disarankan untuk dijalankan
- `human_detect_cli.py`
  - file utama aplikasi
  - berisi deteksi, tracking, dashboard, dan semua logika inti
- `yolov8n.pt`
  - model default untuk backend `ultralytics`
- `README.md`
  - panduan penggunaan project

---

## Cara Kerja Singkat

Secara sederhana, aplikasi bekerja seperti ini:

1. Kamera menangkap frame video
2. Model YOLO mendeteksi objek manusia
3. Sistem tracking memberi `ID` per orang
4. Aplikasi menghitung:
   - berapa orang yang sedang terlihat
   - berapa total orang unik yang pernah muncul
   - status gerak tiap orang
5. Dashboard ditampilkan ke layar atau ke terminal

Kalau kamu hanya ingin langsung pakai, cukup fokus ke bagian:

- instalasi
- jalankan `python human.py`

---

## Instalasi di PC

Bagian ini untuk Windows / Linux / laptop / desktop.

### 1. Clone Repository

```bash
git clone https://github.com/floryid/HumanDetect.git
cd HumanDetect
```

### 2. Pastikan Python Sudah Terpasang

Cek versi Python:

```bash
python --version
```

Disarankan:

- Python `3.10` atau lebih baru

### 3. (Disarankan) Buat Virtual Environment

Tujuan virtual environment: supaya instalasi dependensi rapi dan tidak bentrok dengan project lain.

#### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4. Install Dependensi (PC)

Catatan penting:

- Untuk penggunaan kamera di PC, `opencv-python` wajib (karena source default adalah `opencv`).
- Untuk backend `ultralytics`, package `lap` dibutuhkan untuk tracking agar stabil.
- Gunakan `python -m pip ...` agar menghindari salah Python/pip (terutama di Windows).

#### Opsi termudah untuk PC: backend Ultralytics

```bash
python -m pip install numpy pillow opencv-python ultralytics lap
```

Ini adalah pilihan terbaik jika:

- kamu menjalankan di PC
- ingin setup paling cepat
- ingin pakai file model `yolov8n.pt` yang sudah ada di repo

#### Jika ingin backend ONNX

```bash
python -m pip install numpy pillow opencv-python onnxruntime
```

#### Jika ingin backend TFLite

```bash
python -m pip install numpy pillow opencv-python tflite-runtime
```

### 5. Cek Instalasi (Opsional tapi Disarankan)

Pastikan aplikasi bisa membaca opsi CLI:

```bash
python human.py --help
```

### 6. Jalankan Aplikasi

Perintah paling singkat:

```bash
python human.py
```

Kalau ingin langsung file utama:

```bash
python human_detect_cli.py
```

---

## Instalasi di Termux Android

Bagian ini untuk Android menggunakan Termux.

### Penting Sebelum Mulai

Di Termux, backend yang paling disarankan adalah:

- `onnx`
- `tflite`

Backend `ultralytics` lebih cocok untuk PC karena biasanya lebih mudah dan lebih stabil di sana.

### 1. Install Termux:API (Wajib untuk Kamera)

Karena Termux tidak memakai `opencv` camera langsung, aplikasi mengambil frame menggunakan:

- `termux-camera-photo`

Yang perlu kamu install:

- aplikasi Android: Termux:API (dari F-Droid / Play Store yang tersedia)
- package di Termux: `termux-api`

### 2. Install Package Dasar

```bash
pkg update
pkg install python termux-api
```

### 3. (Disarankan) Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependensi Python (Termux)

Catatan:

- Di Termux, jalankan aplikasi dengan `--source termux-photo` dan biasanya `--no-gui`.
- Untuk mode `termux-photo`, kamu tidak wajib install `opencv-python`.
- `pillow` dipakai untuk baca gambar hasil `termux-camera-photo`.

#### Untuk ONNX

```bash
python -m pip install numpy pillow onnxruntime
```

#### Untuk TFLite

```bash
python -m pip install numpy pillow tflite-runtime
```

### 5. Izin Storage (Opsional)

Kalau kamu mau menyimpan file (mis. model atau output), jalankan:

```bash
termux-setup-storage
```

### 6. Tes Kamera Termux (Opsional tapi Disarankan)

Tes apakah kamera bisa diakses:

```bash
termux-camera-photo -c 0 test.jpg
```

Kalau command ini gagal:

- pastikan Termux:API app terinstall
- pastikan permission kamera sudah diizinkan untuk Termux

### 7. Siapkan Model

Jika kamu memakai Termux, model yang lebih cocok:

- `yolov8n.onnx`
- atau `yolov8n.tflite`

Catatan:

- file default di repo sekarang adalah `yolov8n.pt`
- untuk Termux, biasanya kamu perlu menyiapkan model `.onnx` atau `.tflite` secara terpisah

### 8. Jalankan di Termux

Contoh dengan ONNX:

```bash
python human.py --source termux-photo --backend onnx --model yolov8n.onnx --no-gui
```

Contoh dengan TFLite:

```bash
python human.py --source termux-photo --backend tflite --model yolov8n.tflite --no-gui
```

---

## Cara Menjalankan Aplikasi

Bagian ini menjelaskan perintah yang paling sering dipakai.

### 1. Jalankan Normal

```bash
python human.py
```

Fungsi:

- menjalankan aplikasi dengan pengaturan default
- memakai kamera default
- memakai model default `yolov8n.pt`

### 2. Pilih Kamera Tertentu

```bash
python human.py --camera 0
```

Kalau kamera default tidak terbaca, coba:

```bash
python human.py --camera 1
```

Gunakan ini jika:

- laptop punya lebih dari satu kamera
- webcam eksternal tidak muncul di `camera 0`

### 3. Jalankan Tanpa GUI

```bash
python human.py --no-gui
```

Cocok untuk:

- server
- VPS
- Termux
- monitoring lewat terminal

### 4. Simpan Video Hasil Deteksi

```bash
python human.py --save output.mp4
```

Hasil anotasi akan disimpan ke file video.

### 5. Jalankan dengan Backend Ultralytics

```bash
python human.py --backend ultralytics --model yolov8n.pt
```

Ini adalah mode paling mudah untuk PC.

### 6. Jalankan dengan Backend ONNX

```bash
python human.py --backend onnx --model yolov8n.onnx
```

### 7. Jalankan dengan Backend TFLite

```bash
python human.py --backend tflite --model yolov8n.tflite
```

---

## Penjelasan Backend

Supaya lebih mudah dipahami, berikut arti tiap backend:

### `ultralytics`

Cocok untuk:

- PC / laptop
- pengguna yang ingin setup cepat

Kelebihan:

- paling mudah dipakai
- cocok dengan `yolov8n.pt`
- pengalaman paling nyaman di PC

### `onnx`

Cocok untuk:

- perangkat ringan
- Termux
- deployment yang lebih fleksibel

Kelebihan:

- lebih ringan di banyak perangkat
- cocok untuk mode headless

### `tflite`

Cocok untuk:

- perangkat yang butuh inferensi ringan
- Android / edge device

Kelebihan:

- ringan
- cocok untuk deployment mobile tertentu

Jika bingung memilih:

- pakai `ultralytics` untuk PC
- pakai `onnx` untuk Termux

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
| `--track-iou` | Ambang IoU tracking untuk backend non-ultralytics |
| `--track-max-age` | Batas usia track sebelum dihapus |
| `--termux-interval` | Jeda antar foto di mode Termux |
| `--save` | Simpan video anotasi |

Untuk melihat semua opsi:

```bash
python human.py --help
```

---

## Tampilan Monitoring

Saat aplikasi berjalan, tampilan live akan menunjukkan:

- jumlah orang yang sedang terlihat saat ini
- total orang unik yang terdeteksi selama aplikasi berjalan
- puncak jumlah orang dalam satu sesi
- FPS realtime
- uptime monitoring
- daftar orang aktif
- status tiap orang

Arti status:

- `Dipantau`
  - orang terdeteksi dan sedang dipantau
- `Aktif`
  - orang pernah bergerak dalam sesi berjalan
- `Bergerak`
  - orang sedang terdeteksi berpindah posisi

---

## Contoh Output Terminal

Contoh log terminal:

```text
terlihat_sekarang=2 orang | total_terdeteksi=6 orang | puncak=3 orang | fps=24.8 | uptime=01:23 | orang_baru=+1
 - Orang #1 | akurasi=0.91 | status=Dipantau | pantau=00:17
 - Orang #2 | akurasi=0.88 | status=Bergerak | pantau=00:09
```

Penjelasan:

- `terlihat_sekarang`
  - jumlah orang yang sedang ada di frame sekarang
- `total_terdeteksi`
  - jumlah orang unik selama sesi aplikasi
- `puncak`
  - jumlah orang terbanyak yang pernah terlihat bersamaan
- `fps`
  - kecepatan pemrosesan realtime
- `uptime`
  - lamanya aplikasi berjalan
- `orang_baru`
  - jumlah penambahan orang unik sejak report terakhir

Log ini cocok untuk:

- monitoring tanpa GUI
- debugging
- integrasi ke sistem lain

---

## Command Paling Singkat

Project ini sudah menyediakan launcher singkat:

```bash
python human.py
```

Artinya kamu tidak perlu mengetik file utama yang lebih panjang:

```bash
python human_detect_cli.py
```

Jika tujuanmu hanya ingin menjalankan aplikasi secepat mungkin, cukup pakai:

```bash
python human.py
```

---

## Troubleshooting

### Kamera tidak terbuka

Coba ganti index kamera:

```bash
python human.py --camera 1
```

Kalau masih belum terbuka:

- pastikan kamera tidak dipakai aplikasi lain
- pastikan webcam terdeteksi sistem operasi

### Modul belum terinstall

Install ulang dependensi sesuai backend yang kamu pilih.

Untuk PC dengan `ultralytics`:

```bash
pip install numpy pillow opencv-python ultralytics lap
```

Untuk ONNX:

```bash
pip install numpy pillow opencv-python onnxruntime
```

Untuk TFLite:

```bash
pip install numpy pillow opencv-python tflite-runtime
```

### Ingin lihat semua opsi

```bash
python human.py --help
```

### Di Termux tidak bisa pakai kamera OpenCV

Gunakan:

```bash
--source termux-photo
```

Contoh:

```bash
python human.py --source termux-photo --backend onnx --model yolov8n.onnx --no-gui
```

### Tidak punya model ONNX atau TFLite

Gunakan PC terlebih dahulu dengan backend `ultralytics` dan file:

```text
yolov8n.pt
```

Atau siapkan model:

- `yolov8n.onnx`
- `yolov8n.tflite`

sesuai backend yang ingin dipakai.

---

## Rekomendasi Penggunaan

Untuk hasil terbaik:

- gunakan pencahayaan ruangan yang cukup
- pastikan kamera stabil
- gunakan backend yang sesuai perangkat
- di PC, gunakan `ultralytics`
- di Termux, gunakan `onnx`
- gunakan `--no-gui` jika perangkat terbatas

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

Project ini mengikuti lisensi yang digunakan pada repository GitHub ini.
