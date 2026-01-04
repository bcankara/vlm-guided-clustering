# VLM-Yönlendirmeli Kümeleme - Araştırma Arşivi

Bu arşiv, VLM-Yönlendirmeli Hiyerarşik Kümeleme projesinin tüm deney geçmişini içerir. Buradaki tüm dosyalar `main.py` tarafından üretilen **gerçek process çıktılarıdır** - sonradan işlenmiş akademik figürler içermez.

---

## Arşiv Yapısı

```
archive/
├── phase_0_baby_steps/           # Projenin en erken deneyleri (26 Aralık 2025)
│   ├── ground_truth_data/        # Sentetik ground truth (4 küme)
│   ├── first_vlm_experiment/     # İlk VLM denemesi (K=6)
│   ├── over_split_example/       # Aşırı bölünme örneği (K=30!)
│   ├── kmeans_baseline/          # K-Means baseline sonuçları
│   ├── kshape_baseline/          # K-Shape baseline sonuçları
│   └── fixed_k4/                 # Sabit K=4 deneyi
│
├── phase_1_baseline/             # Erken VLM deneyleri (26-28 Aralık 2025)
│   └── vlm_kshape_early/         # Erken K-Shape VLM çalışması
│
├── phase_2_zone_verification/    # Bölge tabanlı doğrulama (29-31 Aralık 2025)
│   ├── Gemini_3_Flash/           # Gemini 3 Flash ile 17 deney
│   ├── Results_backup/           # 3 ek deney
│   ├── vlm_kshape_20251231_*/    # 2 başarılı K-Shape çalışması
│
└── failed_experiments/           # Aşırı birleştirme hata vakaları
    └── over_merge_cases/         # K=4 yerine K=3
```

---

## Deneyleri Anlamak

### Her Deney Klasöründe Ne Var?

Her `vlm_*` klasörü şunları içerir:

| Dosya/Klasör | Açıklama |
|--------------|----------|
| `step_01/`, `step_02/`, ... | Her VLM karar adımının görselleri ve JSON verileri |
| `final/` | Son küme görselleri ve merge round özetleri |
| `llm_guided/` | Ara sonuçlar (final_results.json, flow_tree.json) |
| `experiment_log.json` | Tam deney kaydı |
| `experiment_report.md` | Otomatik oluşturulan markdown rapor |
| `cluster_hierarchy.txt` | Metin tabanlı küme ağacı |

### Temel Metrikler

- **ARI (Adjusted Rand Index):** Kümeleme doğruluğunu ölçer (1.0 = mükemmel)
- **NMI (Normalized Mutual Information):** Bilgi örtüşmesini ölçer (1.0 = mükemmel)
- **K:** Son küme sayısı (ground truth = 4)

---

## Faz Açıklamaları

### Faz 0: Bebek Adımları (26 Aralık 2025)

Projenin **en erken deneyleri**. Temel öğrenmeler:

- `first_vlm_experiment/`: VLM-yönlendirmeli kümelemenin ilk denemesi (K=6, hafif aşırı bölünme)
- `over_split_example/`: K=30 ile ciddi aşırı bölünme vakası (!)
- Bu faz, ham görsellerin VLM'i karıştırdığını ortaya koydu

**Tipik Sonuçlar:** ARI ~0.30-0.50, K değerleri 6-30

### Faz 1: Baseline (26-28 Aralık 2025)

Düzgün takip ile yapılan ilk yapılandırılmış deneyler:

- Spesifik yönlendirme olmayan temel promptlar
- Görselleştirme normalizasyonu yok
- VLM, genlik farklarını davranış farkları olarak yorumladı

**Tipik Sonuçlar:** ARI ~0.48, K değerleri 10-23

### Faz 2: Bölge Doğrulaması (29-31 Aralık 2025)

Bölge tabanlı doğrulama ile büyük iyileştirmeler:

- x=25, 50, 75, 100, 125 noktalarına dikey referans çizgileri eklendi
- Davranış sınıflandırma sistemi tanıtıldı
- K-Shape algoritması ARI=0.99'a ulaştı

**Tipik Sonuçlar:** ARI ~0.61-0.99, K değerleri 4-8

### Başarısız Deneyler: Aşırı Birleştirme Vakaları

VLM'in kümeleri yanlış birleştirdiği vakalar:

- Ground truth K=4 yerine son K=3
- VLM, yanlış kararlarda %100 güven verdi
- Tüm hatalar Hierarchical algoritmasında

**Ders:** Yüksek güven ≠ doğruluk

---

## Nasıl Tekrarlanır?

1. **Ana deneyi çalıştırın:**
   ```bash
   python main.py
   ```

2. **Deney türünü seçin (menü):**
   - Seçenekler 1-3: Baseline algoritmalar
   - Seçenekler 4-6: VLM-yönlendirmeli algoritmalar
   - Seçenek 7: Sabit K=4 karşılaştırması
   - Seçenek 8: Tekrarlanabilirlik testi (6x çalıştırma)

3. **Sonuçları kontrol edin:**
   - Güncel çalışmalar için `Results/` klasörü
   - Model-spesifik sonuçlar için `Gemini_*/` klasörleri

---

## Dosya Adlandırma Kuralı

Deney klasörleri bu kalıbı takip eder:
```
vlm_{algoritma}_{YYYYAAGG}_{SSDDSS}
```

Örnek: `vlm_kmeans_20251229_185203`
- Algoritma: K-Means
- Tarih: 29 Aralık 2025
- Saat: 18:52:03

---

## İletişim

**Dr. Burak Can KARA**  
Amasya Üniversitesi

- E-posta: burakcankara@gmail.com
- Web sitesi: https://bcankara.com
- Projeler: https://deformationdb.com | https://insar.tr

---

*Arşiv oluşturulma tarihi: 4 Ocak 2026*
