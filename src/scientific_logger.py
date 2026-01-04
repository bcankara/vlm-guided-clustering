"""
Bilimsel Loglama Modülü
Akademik makale için tüm adımları detaylı kaydeder

Kayıt edilenler:
- Veri yükleme istatistikleri
- Kümeleme parametreleri ve sonuçları
- K değeri belirleme süreci
- Metrik hesaplamaları
- Zaman damgaları
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """NumPy tiplerini JSON'a çeviren encoder"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ScientificLogger:
    """
    Bilimsel çalışma için kapsamlı loglama sınıfı.
    
    Hem console'a hem dosyaya hem de JSON formatında yapılandırılmış log tutar.
    """
    
    def __init__(self, experiment_name: str = "sar_clustering"):
        """
        Args:
            experiment_name: Deney adı (dosya isimleri için)
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Log dizini
        self.log_dir = Path("outputs/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Dosya isimleri
        self.log_file = self.log_dir / f"{experiment_name}_{self.timestamp}.log"
        self.json_file = self.log_dir / f"{experiment_name}_{self.timestamp}.json"
        
        # Yapılandırılmış log verisi
        self.experiment_data = {
            "experiment_name": experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "steps": [],
            "data": {},
            "clustering": {},
            "metrics": {},
            "summary": {}
        }
        
        # Python logger kurulumu
        self._setup_logger()
        
        self.info(f"{'='*60}")
        self.info(f"DENEY BAŞLATILDI: {experiment_name}")
        self.info(f"Zaman: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Log dosyası: {self.log_file}")
        self.info(f"{'='*60}")
    
    def _setup_logger(self):
        """Python logger kurulumu"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Önceki handler'ları temizle
        self.logger.handlers = []
        
        # Dosya handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Bilgi mesajı"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Debug mesajı (sadece dosyaya)"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Uyarı mesajı"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Hata mesajı"""
        self.logger.error(message)
    
    def step(self, step_name: str, step_number: int, total_steps: int):
        """Yeni adım başlat"""
        self.current_step = {
            "step_number": step_number,
            "step_name": step_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "details": {}
        }
        self.info(f"\n[{step_number}/{total_steps}] {step_name}")
        self.info("-" * 50)
    
    def step_complete(self, details: Optional[Dict] = None):
        """Adımı tamamla"""
        if hasattr(self, 'current_step'):
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.current_step["start_time"])
            duration = (end_time - start_time).total_seconds()
            
            self.current_step["end_time"] = end_time.isoformat()
            self.current_step["duration_seconds"] = duration
            if details:
                self.current_step["details"] = details
            
            self.experiment_data["steps"].append(self.current_step)
            self.info(f"✓ Tamamlandı ({duration:.2f} saniye)")
    
    def log_data_info(self, info: Dict):
        """Veri bilgilerini kaydet"""
        self.experiment_data["data"] = info
        
        self.info(f"Veri Özeti:")
        for key, value in info.items():
            self.info(f"  • {key}: {value}")
    
    def log_k_search(self, algorithm: str, k: int, score: float, score_name: str = "silhouette"):
        """K arama sürecini kaydet"""
        if algorithm not in self.experiment_data["clustering"]:
            self.experiment_data["clustering"][algorithm] = {
                "k_search": [],
                "optimal_k": None,
                "final_result": None
            }
        
        self.experiment_data["clustering"][algorithm]["k_search"].append({
            "k": k,
            score_name: score
        })
        
        self.debug(f"  {algorithm} k={k}: {score_name}={score:.4f}")
    
    def log_optimal_k(self, algorithm: str, optimal_k: int, best_score: float, score_name: str = "silhouette"):
        """Optimal K değerini kaydet"""
        if algorithm in self.experiment_data["clustering"]:
            self.experiment_data["clustering"][algorithm]["optimal_k"] = optimal_k
            self.experiment_data["clustering"][algorithm]["best_score"] = best_score
            self.experiment_data["clustering"][algorithm]["score_metric"] = score_name
        
        self.info(f"  {algorithm}: Optimal k={optimal_k} (en iyi {score_name}={best_score:.4f})")
    
    def log_clustering_result(self, algorithm: str, result: Dict):
        """Kümeleme sonuçlarını kaydet"""
        if algorithm not in self.experiment_data["clustering"]:
            self.experiment_data["clustering"][algorithm] = {}
        
        # Labels hariç kaydet (çok büyük)
        clean_result = {k: v for k, v in result.items() if k not in ['labels', 'cluster_centers']}
        
        # Küme boyutlarını hesapla
        if 'labels' in result:
            labels = result['labels']
            unique_labels = set(labels)
            cluster_sizes = {str(l): int(np.sum(labels == l)) for l in unique_labels if l != -1}
            clean_result['cluster_sizes'] = cluster_sizes
            if -1 in unique_labels:
                clean_result['noise_points'] = int(np.sum(labels == -1))
        
        self.experiment_data["clustering"][algorithm]["final_result"] = clean_result
        
        self.info(f"  {algorithm} Sonucu:")
        self.info(f"    • Bulunan küme sayısı: {result.get('n_clusters_found', 'N/A')}")
        if 'cluster_sizes' in clean_result:
            for label, size in clean_result['cluster_sizes'].items():
                self.info(f"    • Küme {label}: {size} nokta")
    
    def log_metrics(self, algorithm: str, metrics: Dict):
        """Değerlendirme metriklerini kaydet"""
        if algorithm not in self.experiment_data["metrics"]:
            self.experiment_data["metrics"][algorithm] = {}
        
        self.experiment_data["metrics"][algorithm] = metrics
        
        self.info(f"  {algorithm} Metrikleri:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"    • {metric}: {value:.4f}")
            else:
                self.info(f"    • {metric}: {value}")
    
    def log_comparison(self, comparison_data: Dict):
        """Algoritma karşılaştırma sonuçlarını kaydet"""
        self.experiment_data["summary"]["comparison"] = comparison_data
    
    def finalize(self, summary: Optional[Dict] = None):
        """Deneyi tamamla ve tüm logları kaydet"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.experiment_data["end_time"] = end_time.isoformat()
        self.experiment_data["duration_seconds"] = duration
        
        if summary:
            self.experiment_data["summary"].update(summary)
        
        # JSON dosyasına kaydet
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        self.info(f"\n{'='*60}")
        self.info(f"DENEY TAMAMLANDI")
        self.info(f"Toplam süre: {duration:.2f} saniye ({duration/60:.2f} dakika)")
        self.info(f"Log dosyası: {self.log_file}")
        self.info(f"JSON dosyası: {self.json_file}")
        self.info(f"{'='*60}")
        
        return self.experiment_data
    
    def get_experiment_data(self) -> Dict:
        """Deney verilerini döndür"""
        return self.experiment_data


# Global logger instance
_logger: Optional[ScientificLogger] = None


def get_logger(experiment_name: str = "sar_clustering") -> ScientificLogger:
    """Global logger instance al veya oluştur"""
    global _logger
    if _logger is None:
        _logger = ScientificLogger(experiment_name)
    return _logger


def reset_logger():
    """Logger'ı sıfırla (yeni deney için)"""
    global _logger
    _logger = None
