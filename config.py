"""
LOS Deformasyon Kümeleme Projesi - Konfigürasyon

Author: Dr. Burak Can KARA
Affiliation: Amasya University
Email: burakcankara@gmail.com
Website: https://bcankara.com
"""
import os
from pathlib import Path

# Proje dizinleri
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# NOT: Dizinler main.py'de dinamik olarak oluşturulacak

# SAR Veri Konfigürasyonu (Gerçek Veri)
SAR_CONFIG = {
    'nc_file': 'sar_combined_data.nc',
    'min_velocity': 20.0,           # Minimum mutlak düşey hız (mm/year)
    'use_vertical': True,           # Düşey deformasyon kullan
    'deformation_var': 'los_ver',   # Düşey LOS deformasyon değişkeni
    'velocity_var': 'vel_ver',      # Düşey hız değişkeni
}

# Sentetik Veri parametreleri (eski, referans için)
DATA_CONFIG = {
    'n_points': 10000,
    'n_timesteps': 182,
    'timestep_days': 12,
    'n_clusters': 4,
    'min_points_per_cluster': 1500,
    'lat_range': (39.0, 41.0),
    'lon_range': (32.0, 35.0),
    'random_seed': 42
}

# Kümeleme parametreleri
CLUSTERING_CONFIG = {
    'algorithms': ['kmeans_auto', 'kshape_auto', 'hdbscan'],
    # K-Means ve K-Shape için arama aralığı
    'k_range': (2, 10),
    # HDBSCAN parametreleri (otomatik küme sayısı)
    'hdbscan_min_cluster_size': 100,
    'hdbscan_min_samples': 10
}

# Gemini API
GEMINI_CONFIG = {
    'model': 'gemini-2.5-pro'  # API key is managed in settings.json
}

# Değerlendirme metrikleri
EVALUATION_METRICS = ['ARI', 'NMI', 'Purity', 'Silhouette', 'n_clusters_found']
