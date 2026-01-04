<p align="center">
  <img src="docs/logo.png" alt="VLM-Guided Clustering Logo" width="150" height="150">
</p>

<h1 align="center">VLM-Guided Hierarchical Clustering</h1>

<p align="center">
  <strong>Vision-Language Model powered time series clustering for InSAR deformation analysis</strong>
</p>

<p align="center">
  <a href="#features"><img src="https://img.shields.io/badge/AI-Gemini%20VLM-blue?style=for-the-badge&logo=google" alt="Gemini VLM"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License"></a>
  <a href="https://bcankara.com"><img src="https://img.shields.io/badge/Author-Dr.%20Burak%20Can%20KARA-purple?style=for-the-badge" alt="Author"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#gemini-prompts">Gemini Prompts</a> •
  <a href="#results">Results</a> •
  <a href="#contact">Contact</a>
</p>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Analysis** | Google Gemini VLM analyzes cluster visualizations for homogeneity |
| 📊 **Multi-Algorithm** | Supports K-Means, K-Shape, and Hierarchical clustering |
| 🔄 **Iterative Refinement** | Split → Analyze → Merge workflow for optimal clusters |
| 📈 **Academic Metrics** | ARI, NMI comparison against ground truth |
| 📝 **Auto-Logging** | Detailed JSON/Markdown experiment reports |
| 🔬 **Reproducibility** | Built-in reproducibility testing framework |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bcankara/vlm-guided-clustering.git
cd vlm-guided-clustering

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `settings.json` and add your Gemini API key:

```json
{
  "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
  "gemini_model": "gemini-2.5-pro",
  "k_range": [2, 8],
  "min_cluster_size": 25,
  "merge_viz_mode": "v2"
}
```

> 💡 Get a free API key from [Google AI Studio](https://aistudio.google.com/)

### Run

```bash
python main.py
```

---

## 🧬 How It Works

<p align="center">
  <img src="docs/vlm_clustering_diagram.png" alt="VLM Clustering Workflow" width="700">
</p>

### Workflow

```
┌─────────────────┐
│   Time Series   │
│     Data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Initial K-Means │
│   / K-Shape     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Cluster Queue  │────▶│    Gemini VLM    │
│                 │     │ (16-sample grid) │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │              ┌────────┴────────┐
         │              ▼                 ▼
         │      Homogeneous?       Heterogeneous?
         │          │                     │
         │          ▼                     ▼
         │      ❄️ FREEZE            ✂️ SPLIT
         │                          (try K=2,3,4)
         │                               │
         ▼                               ▼
┌─────────────────┐           ┌─────────────────┐
│   Merge Phase   │◀──────────│   Re-queue      │
│ (batch-wise 3)  │           │   sub-clusters  │
└────────┬────────┘           └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Clusters │
│   (ARI, NMI)    │
└─────────────────┘
```

---

## 🎯 Gemini Prompts

The system uses 4 specialized prompts for VLM analysis:

### 1. Homogeneity Analysis

**Purpose**: Determines if all time series in a cluster share the same physical behavior.

**Visual Input**: 4×4 grid (16 sample time series sorted by slope)

```
CRITICAL CHECK:
1. If trends are OPPOSITE (Up vs Down) → SPLIT immediately
2. If shapes are DIFFERENT (Wave vs Straight) → SPLIT
3. If peaks/valleys do NOT align in time → SPLIT only if shifts are large
4. Only if indistinguishable → HOMOGENEOUS
```

**Output**:
```json
{
    "is_homogeneous": true/false,
    "should_split": true/false,
    "distinct_groups": <number>,
    "confidence": <0-100>
}
```

---

### 2. Self-Correction (Reflexion)

**Purpose**: Reduces false positive SPLIT decisions by requesting a second evaluation.

**Triggered when**: Initial SPLIT decision has confidence < 80%

```
Remember:
- Minor noise differences are NORMAL
- Small phase shifts are ACCEPTABLE
- Only split if there is UNDENIABLE proof
```

---

### 3. Merge V1 (Overlay)

**Purpose**: Identifies clusters that are the same signal at different amplitudes.

**Visual Input**: All cluster means overlaid on the same axes

```
"TRAIN TRACKS" TEST: 
- If lines run PARALLEL → MERGE (same signal, different scale)
- If they CROSS each other (X-shape) → DO NOT MERGE

PEAK/VALLEY ALIGNMENT:
- Do peaks occur at the exact same X-position?
```

---

### 4. Merge V2 (Subplot) — *Default*

**Purpose**: More precise merge decisions focusing on seasonal wave amplitude differences.

**Visual Input**: Each cluster in separate subplot with shared Y-axis scale

```
CRITICAL DISTINCTION:
1. LINEAR TREND DEPTH: Differences are OK! 
   (one goes -60mm, another -120mm = SAME behavior)

2. SEASONAL WAVE AMPLITUDE: Differences are NOT OK!
   (one has 10mm waves, another 40mm waves = DIFFERENT behavior)

MERGE IF:
✅ Same wave pattern (peaks at same times)
✅ Similar seasonal amplitude RELATIVE to each other
✅ No flat sections while others move

DO NOT MERGE IF:
❌ One has strong waves, another has weak/no waves
❌ One has a FLAT section while others continue moving
❌ Opposite overall trends
```

---

## 📊 Results

Performance comparison on synthetic InSAR data (Ground Truth: K=4):

| Algorithm | Found K | ARI | NMI |
|-----------|:-------:|:---:|:---:|
| K-Means + VLM | 4 | **0.95** | 0.93 |
| K-Shape + VLM | 4 | **0.99** | 0.98 |
| Hierarchical + VLM | 4 | **0.92** | 0.90 |

> **ARI** = Adjusted Rand Index, **NMI** = Normalized Mutual Information

---

## 📁 Project Structure

```
.
├── main.py                 # Main application
├── generate_data.py        # Synthetic data generator
├── run_reproducibility.py  # Reproducibility testing
├── config.py               # Configuration constants
├── settings.json           # User settings (API key here)
├── requirements.txt        # Dependencies
└── src/
    ├── settings.py         # Settings management
    ├── tracker.py          # Experiment tracker
    └── scientific_logger.py # Scientific logging
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

<p align="center">
  <strong>Dr. Burak Can KARA</strong><br>
  Amasya University
</p>

<p align="center">
  <a href="mailto:burakcankara@gmail.com"><img src="https://img.shields.io/badge/Email-burakcankara%40gmail.com-red?style=flat-square&logo=gmail" alt="Email"></a>
  <a href="https://bcankara.com"><img src="https://img.shields.io/badge/Website-bcankara.com-blue?style=flat-square&logo=safari" alt="Website"></a>
  <a href="https://deformationdb.com"><img src="https://img.shields.io/badge/Project-DeformationDB-green?style=flat-square&logo=satellite" alt="DeformationDB"></a>
  <a href="https://insar.tr"><img src="https://img.shields.io/badge/Project-InSAR.tr-purple?style=flat-square&logo=satellite" alt="InSAR.tr"></a>
</p>

---

<p align="center">
  <sub>Built with ❤️ for InSAR time series analysis research</sub>
</p>
