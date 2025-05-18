# Fake News Optimizer

Proyek ini menganalisis dan mengoptimalkan kumpulan data berita palsu menggunakan berbagai algoritma pembelajaran mesin untuk meningkatkan deteksi berita palsu.

## Deskripsi

Proyek ini mengimplementasikan tiga pendekatan berbeda untuk deteksi berita palsu:

1. **Graph Neural Network (GNN)**: Menggunakan jaringan saraf graf untuk memodelkan hubungan antara kata-kata dan fitur dalam teks.
2. **Electric EEL (Enhanced Evolutionary Learning)**: Algoritma evolusioner yang terinspirasi oleh perilaku belut listrik untuk seleksi fitur dan optimasi.
3. **Hybrid GNN-EEL**: Model hibrida yang menggabungkan seleksi fitur dari algoritma Electric EEL dengan kekuatan pemodelan GNN.

## Dataset

Proyek ini menggunakan dataset dari PolitiFact dan GossipCop yang berisi berita asli dan palsu. File data disusun dalam:

- `politifact_real.csv`: Berita asli dari PolitiFact
- `politifact_fake.csv`: Berita palsu dari PolitiFact
- `gossipcop_real.csv`: Berita asli dari GossipCop
- `gossipcop_fake.csv`: Berita palsu dari GossipCop

## Kebutuhan Sistem

Untuk menjalankan proyek ini, Anda memerlukan Python 3.8 atau lebih tinggi dan pustaka-pustaka berikut:

```
pip install -r requirements.txt
```

## Struktur Proyek

- `combine_datasets.py`: Skrip untuk menggabungkan berbagai kumpulan data menjadi satu
- `gnn_optimization.py`: Implementasi model GNN
- `electric_eel_optimization.py`: Implementasi algoritma Electric EEL
- `hybrid_gnn_eel_optimization.py`: Implementasi model hibrida GNN + Electric EEL
- `compare_models.py`: Skrip untuk membandingkan kinerja ketiga model
- `requirements.txt`: Daftar ketergantungan proyek

## Cara Menjalankan

1. Pertama, gabungkan dataset:

```
python combine_datasets.py
```

2. Jalankan setiap model optimasi:

```
python gnn_optimization.py
python electric_eel_optimization.py
python hybrid_gnn_eel_optimization.py
```

3. Bandingkan hasilnya:

```
python compare_models.py
```

Ini akan menghasilkan tabel perbandingan dan visualisasi grafik untuk membandingkan kinerja ketiga model.

## Hasil

Hasil dari setiap model disimpan dalam file CSV terpisah:
- `gnn_results.csv`
- `electric_eel_results.csv`
- `hybrid_gnn_eel_results.csv`

Skrip perbandingan menghasilkan:
- Tabel perbandingan (`model_comparison_table.txt`)
- Grafik batang metrik (`model_comparison_metrics.png`)
- Perbandingan waktu pemrosesan (`model_comparison_time.png`)
- Grafik radar untuk perbandingan visual (`model_comparison_radar.png`)

## Metrik Evaluasi

Model-model dievaluasi menggunakan:
- Akurasi (Accuracy)
- Presisi (Precision)
- Recall
- Skor F1 (F1 Score)

## Fitur Tambahan

- Algoritma Electric EEL menerapkan seleksi fitur untuk mengurangi dimensionalitas
- Model hibrida GNN-EEL menggunakan Electric EEL untuk mengoptimalkan fitur sebelum melatih GNN
- Semua model menyertakan pra-pemrosesan teks untuk membersihkan dan menormalkan konten berita

## Penulis

Riski Yuniar

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT - lihat file LICENSE untuk detail lebih lanjut. 
