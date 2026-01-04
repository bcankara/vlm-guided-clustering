"""
Sentetik SAR Verisi Üretici
Gerçekçi deformasyon desenleri ile 4 küme

Küme Tipleri:
A: Monoton Çökme - Düz iniş
B: Mevsimsel Toparlanmalı - Çökme + yaz toparlanması  
C: Periyodik Hızlı/Stabil - 1 yıl hızlı, 1 yıl yavaş
D: Stabilize Olan - Başta hızlı, sonra yavaşlama

Author: Dr. Burak Can KARA
Affiliation: Amasya University
Email: burakcankara@gmail.com
Website: https://bcankara.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta
import json

OUTPUT_DIR = Path("outputs/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SyntheticDataGenerator:
    """Gerçekçi sentetik SAR verisi üretici"""
    
    def __init__(
        self,
        n_points: int = 10000,
        n_years: float = 5.0,
        step_days: int = 12,
        cluster_ratios: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        random_seed: int = 42
    ):
        self.n_points = n_points
        self.n_years = n_years
        self.step_days = step_days
        self.n_timesteps = int(n_years * 365 / step_days)
        self.cluster_ratios = cluster_ratios
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        self.t = np.arange(self.n_timesteps)
        self.t_years = self.t * step_days / 365
        
        self.cluster_names = {
            0: 'Monoton Çökme',
            1: 'Mevsimsel Toparlanmalı',
            2: 'Periyodik Hızlı/Stabil',
            3: 'Stabilize Olan'
        }
    
    def _add_noise(self, signal: np.ndarray, level: float = 3.0) -> np.ndarray:
        """Düşük seviye gerçekçi gürültü"""
        n = len(signal)
        white = np.random.normal(0, level, n)
        red = np.zeros(n)
        for i in range(1, n):
            red[i] = 0.6 * red[i-1] + np.random.normal(0, level * 0.2)
        return signal + white + red
    
    def _generate_monotonic(self, n: int) -> np.ndarray:
        """Küme A: Monoton Çökme"""
        data = np.zeros((n, self.n_timesteps))
        for i in range(n):
            rate = np.random.uniform(-70, -30)
            trend = rate * self.t_years
            seasonal = np.random.uniform(2, 6) * np.sin(2 * np.pi * self.t_years)
            data[i] = self._add_noise(trend + seasonal, np.random.uniform(2, 4))
        return data
    
    def _generate_seasonal(self, n: int) -> np.ndarray:
        """Küme B: Mevsimsel Toparlanmalı"""
        data = np.zeros((n, self.n_timesteps))
        for i in range(n):
            rate = np.random.uniform(-50, -25)
            trend = rate * self.t_years
            amp = np.random.uniform(15, 35)
            phase = np.random.uniform(0, 0.5)
            seasonal = amp * np.sin(2 * np.pi * self.t_years + phase)
            half = np.random.uniform(3, 8) * np.sin(4 * np.pi * self.t_years)
            data[i] = self._add_noise(trend + seasonal + half, np.random.uniform(2, 4))
        return data
    
    def _generate_periodic(self, n: int) -> np.ndarray:
        """Küme C: Periyodik Hızlı/Stabil"""
        data = np.zeros((n, self.n_timesteps))
        steps_per_year = int(365 / self.step_days)
        
        for i in range(n):
            signal = np.zeros(self.n_timesteps)
            current = 0
            fast = np.random.uniform(-60, -35)
            slow = np.random.uniform(-10, 5)
            start_fast = True
            
            for year in range(int(self.n_years) + 1):
                s = year * steps_per_year
                e = min((year + 1) * steps_per_year, self.n_timesteps)
                if s >= self.n_timesteps:
                    break
                rate = fast if (year % 2 == 0) == start_fast else slow
                seg = np.linspace(0, rate * (e-s) * self.step_days / 365, e-s)
                signal[s:e] = current + seg
                current = signal[e-1]
            
            seasonal = np.random.uniform(3, 8) * np.sin(2 * np.pi * self.t_years)
            data[i] = self._add_noise(signal + seasonal, np.random.uniform(2, 4))
        return data
    
    def _generate_stabilizing(self, n: int) -> np.ndarray:
        """Küme D: Stabilize Olan"""
        data = np.zeros((n, self.n_timesteps))
        for i in range(n):
            init_rate = np.random.uniform(-80, -40)
            decay = np.random.uniform(0.3, 0.8)
            max_def = init_rate * self.n_years * 0.6
            trend = max_def * (1 - np.exp(-decay * self.t_years))
            seasonal = np.random.uniform(5, 15) * np.sin(2 * np.pi * self.t_years)
            data[i] = self._add_noise(trend + seasonal, np.random.uniform(2, 4))
        return data
    
    def generate(self) -> Tuple[pd.DataFrame, list, np.ndarray]:
        """Veri üret"""
        sizes = [int(self.n_points * r) for r in self.cluster_ratios]
        sizes[-1] = self.n_points - sum(sizes[:-1])
        
        print(f"Sentetik veri üretiliyor: {self.n_points} nokta, {self.n_timesteps} adım")
        
        generators = [self._generate_monotonic, self._generate_seasonal,
                      self._generate_periodic, self._generate_stabilizing]
        
        data_parts, label_parts = [], []
        for cid, (gen, n) in enumerate(zip(generators, sizes)):
            data_parts.append(gen(n))
            label_parts.append(np.full(n, cid))
            print(f"  Küme {cid} ({self.cluster_names[cid]}): {n} nokta")
        
        all_data = np.vstack(data_parts)
        labels = np.concatenate(label_parts)
        
        # Karıştır
        idx = np.random.permutation(self.n_points)
        all_data, labels = all_data[idx], labels[idx]
        
        # DataFrame
        ts_cols = [f"t_{i:03d}" for i in range(self.n_timesteps)]
        ts_df = pd.DataFrame(all_data, columns=ts_cols)
        meta_df = pd.DataFrame({
            'latitude': np.random.uniform(38, 39, self.n_points),
            'longitude': np.random.uniform(28, 29, self.n_points),
            'ground_truth': labels
        })
        df = pd.concat([meta_df, ts_df], axis=1)
        
        return df, ts_cols, labels
    
    def save(self, df: pd.DataFrame, ts_cols: list, labels: np.ndarray):
        """Kaydet ve görselleştir"""
        df.to_csv(OUTPUT_DIR / "synthetic_data.csv", index=False)
        np.save(OUTPUT_DIR / "ground_truth.npy", labels)
        
        meta = {
            'n_points': self.n_points,
            'n_timesteps': self.n_timesteps,
            'n_years': self.n_years,
            'step_days': self.step_days,
            'cluster_names': self.cluster_names,
            'cluster_sizes': {str(i): int((labels==i).sum()) for i in range(4)}
        }
        with open(OUTPUT_DIR / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"Veriler kaydedildi: {OUTPUT_DIR}")
        
        # Otomatik görselleştirme
        self._visualize_clusters(df, ts_cols, labels)
    
    def _visualize_clusters(self, df: pd.DataFrame, ts_cols: list, labels: np.ndarray):
        """Kümeleri görselleştir ve kaydet"""
        timeseries = df[ts_cols].values
        n_timesteps = len(ts_cols)
        dates = [datetime(2020, 1, 1) + timedelta(days=i*self.step_days) for i in range(n_timesteps)]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for cluster_id in range(4):
            ax = axes[cluster_id]
            mask = labels == cluster_id
            cluster_ts = timeseries[mask]
            
            # 30 örnek çiz
            n_samples = min(30, len(cluster_ts))
            sample_idx = np.random.choice(len(cluster_ts), n_samples, replace=False)
            for i in sample_idx:
                ax.plot(dates, cluster_ts[i], alpha=0.3, lw=0.8, color=colors[cluster_id])
            
            # Ortalama
            mean_ts = cluster_ts.mean(axis=0)
            ax.plot(dates, mean_ts, 'k-', lw=2.5, label='Ortalama')
            
            # Başlık ve istatistikler
            total_change = mean_ts[-1] - mean_ts[0]
            slope = np.polyfit(range(len(mean_ts)), mean_ts, 1)[0]
            ax.set_title(f'Küme {cluster_id}: {self.cluster_names[cluster_id]}\n({len(cluster_ts)} nokta)', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Tarih')
            ax.set_ylabel('LOS Deformasyon (mm)')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
            ax.text(0.98, 0.02, f'Toplam: {total_change:.0f} mm\nEğim: {slope:.3f} mm/step', 
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('GROUND TRUTH KÜMELERİ', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ground_truth_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Görsel kaydedildi: {OUTPUT_DIR / 'ground_truth_visualization.png'}")
        
        # Özet tablo
        print("\n" + "="*60)
        print(" GROUND TRUTH ÖZET")
        print("="*60)
        print(f"{'Küme':<35} {'N':>6} {'Toplam':>10} {'Eğim':>10}")
        print("-"*60)
        for cluster_id in range(4):
            mask = labels == cluster_id
            cluster_ts = timeseries[mask]
            mean_ts = cluster_ts.mean(axis=0)
            total_change = mean_ts[-1] - mean_ts[0]
            slope = np.polyfit(range(len(mean_ts)), mean_ts, 1)[0]
            print(f"{cluster_id}: {self.cluster_names[cluster_id]:<30} {mask.sum():>6} {total_change:>10.1f} {slope:>10.3f}")
        print("="*60)


if __name__ == "__main__":
    gen = SyntheticDataGenerator(n_points=10000, n_years=5.0, random_seed=42)
    df, ts_cols, labels = gen.generate()
    gen.save(df, ts_cols, labels)
