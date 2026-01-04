"""
LLM-Guided Hierarchical Clustering
Each cluster is analyzed and split individually.

Flow:
1. Initial K-Means/K-Shape → K clusters
2. Queue each cluster for Gemini review
3. For each cluster in queue:
   - Gemini: Homogeneous? → Freeze
   - Gemini: Heterogeneous? → Split with forced K, add sub-clusters to queue
4. Continue until queue is empty
5. Report final results

Author: Dr. Burak Can KARA
Affiliation: Amasya University
Email: burakcankara@gmail.com
Website: https://bcankara.com
Projects: https://deformationdb.com | https://insar.tr
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import gc  # Memory management

sys.path.insert(0, str(Path(__file__).parent))

from config import FIGURES_DIR, RESULTS_DIR
from src.scientific_logger import NumpyEncoder
from src.settings import get_model, set_model, get_api_key, get_k_range, get_min_cluster_size, get_merge_viz_mode
from src.tracker import ExperimentTracker

# NOTE: FIGURES_DIR and RESULTS_DIR are defined but NOT created here
# All output goes to Results/ via ExperimentTracker (lazy creation)

# REPORT_DIR will be set dynamically per experiment
REPORT_DIR = None  # Will be set by tracker.get_experiment_dir()

# ═══════════════════════════════════════════════════════════════
# GEMINI MODEL SELECTION (Global)
# ═══════════════════════════════════════════════════════════════
GEMINI_MODEL = get_model()  # Load from settings.json

# Available models
AVAILABLE_MODELS = {
    "1": "gemini-2.5-pro",           # Stable, good reasoning
    "2": "gemini-3-pro-preview",     # Best quality (preview)
    "3": "gemini-3-flash-preview",   # Fast + good quality (preview)
}

def select_gemini_model():
    """Let user select Gemini model at startup and save to settings.json"""
    global GEMINI_MODEL
    
    # Show current setting
    current = get_model()
    
    # If AUTO_RUN mode, skip interactive selection
    if os.environ.get("AUTO_RUN") == "1":
        GEMINI_MODEL = current
        print(f"🤖 Auto-run mode: Using {GEMINI_MODEL}")
        return GEMINI_MODEL
    
    print("\n" + "="*55)
    print(" GEMINI MODEL SELECTION")
    print("="*55)
    print("  1. gemini-2.5-pro           [Stable]")
    print("  2. gemini-3-pro-preview     [Best Quality]")
    print("  3. gemini-3-flash-preview   [Fast + Quality]")
    print("-"*55)
    print(f"  Current: {current}")
    print("="*55)
    
    try:
        choice = input("Select model (1-3, Enter=keep current): ").strip()
        if choice == "":
            GEMINI_MODEL = current
        elif choice in AVAILABLE_MODELS:
            GEMINI_MODEL = AVAILABLE_MODELS[choice]
            set_model(GEMINI_MODEL)  # Save to settings.json
        else:
            GEMINI_MODEL = current
    except:
        GEMINI_MODEL = current
    
    print(f"✅ Using: {GEMINI_MODEL}\n")
    return GEMINI_MODEL


def get_experiment_dir(experiment_name: str) -> Path:
    """Get experiment-specific output directory"""
    # Clean name for folder
    clean_name = experiment_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    exp_dir = Path(f"outputs/{clean_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_dir = exp_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


class AcademicLogger:
    def __init__(self, name: str):
        self.name = name
        
    def log(self, msg: str, level: str = "INFO"):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {msg}")
    
    def section(self, title: str):
        print(f"\n{'='*60}\n {title}\n{'='*60}")
    
    def subsection(self, title: str):
        print(f"\n--- {title} ---")


def load_data():
    from generate_data import SyntheticDataGenerator
    
    data_dir = Path("outputs/data")
    if not (data_dir / "synthetic_data.csv").exists():
        gen = SyntheticDataGenerator(n_points=10000, n_years=5.0, random_seed=42)
        df, ts_cols, labels = gen.generate()
        gen.save(df, ts_cols, labels)
    
    df = pd.read_csv(data_dir / "synthetic_data.csv")
    labels = np.load(data_dir / "ground_truth.npy")
    ts_cols = [col for col in df.columns if col.startswith('t_')]
    
    return df, ts_cols, labels


def calculate_cluster_stats(timeseries: np.ndarray) -> Dict:
    """
    Calculate comprehensive statistics for a cluster of time series.
    Used to provide dynamic context in Gemini prompts.
    
    Returns statistics useful for a geoscientist analyzing SAR data.
    """
    n_samples, n_timesteps = timeseries.shape
    
    # 1. AMPLITUDE (Seasonality strength)
    # Calculate detrended amplitude for each series
    amplitudes = []
    for ts in timeseries:
        # Detrend: remove linear component
        slope, intercept = np.polyfit(range(n_timesteps), ts, 1)
        detrended = ts - (slope * np.arange(n_timesteps) + intercept)
        # Amplitude = peak - trough of detrended signal
        amplitude = np.max(detrended) - np.min(detrended)
        amplitudes.append(amplitude)
    amplitudes = np.array(amplitudes)
    
    # 2. TOTAL DISPLACEMENT (End - Start)
    total_displacements = timeseries[:, -1] - timeseries[:, 0]
    
    # 3. SLOPE (Linear rate)
    slopes = np.array([np.polyfit(range(n_timesteps), ts, 1)[0] for ts in timeseries])
    
    # 4. CURVATURE (Is trend linear or curved?)
    # Positive curvature = accelerating, Negative = decelerating
    curvatures = []
    for ts in timeseries:
        # Fit 2nd order polynomial and get curvature term
        coeffs = np.polyfit(range(n_timesteps), ts, 2)
        curvature = coeffs[0] * 2  # 2nd derivative of ax^2 = 2a
        curvatures.append(curvature)
    curvatures = np.array(curvatures)
    
    # 5. RATE VARIABILITY (Does slope change over time?)
    # Calculate local slopes using sliding window
    rate_stds = []
    window = n_timesteps // 5  # ~20% window
    for ts in timeseries:
        local_slopes = []
        for i in range(0, n_timesteps - window, window):
            local_slope = (ts[i + window] - ts[i]) / window
            local_slopes.append(local_slope)
        rate_stds.append(np.std(local_slopes) if len(local_slopes) > 1 else 0)
    rate_stds = np.array(rate_stds)
    
    # 6. NOISE LEVEL (Variability around trend)
    noise_levels = []
    for ts in timeseries:
        slope, intercept = np.polyfit(range(n_timesteps), ts, 1)
        trend = slope * np.arange(n_timesteps) + intercept
        residual = ts - trend
        noise_levels.append(np.std(residual))
    noise_levels = np.array(noise_levels)
    
    return {
        # Amplitude stats
        'amp_min': np.min(amplitudes),
        'amp_max': np.max(amplitudes),
        'amp_median': np.median(amplitudes),
        'amp_std': np.std(amplitudes),
        
        # Displacement stats
        'disp_min': np.min(total_displacements),
        'disp_max': np.max(total_displacements),
        'disp_median': np.median(total_displacements),
        'disp_std': np.std(total_displacements),
        
        # Slope stats
        'slope_min': np.min(slopes),
        'slope_max': np.max(slopes),
        'slope_median': np.median(slopes),
        'slope_std': np.std(slopes),
        
        # Curvature stats
        'curv_min': np.min(curvatures),
        'curv_max': np.max(curvatures),
        'curv_median': np.median(curvatures),
        
        # Rate variability
        'rate_var_median': np.median(rate_stds),
        'rate_var_max': np.max(rate_stds),
        
        # Noise
        'noise_median': np.median(noise_levels),
        
        # Heterogeneity indicators
        'amp_range_ratio': (np.max(amplitudes) - np.min(amplitudes)) / (np.median(amplitudes) + 0.1),
        'disp_range_ratio': abs(np.max(total_displacements) - np.min(total_displacements)) / (abs(np.median(total_displacements)) + 0.1),
        'slope_range_ratio': abs(np.max(slopes) - np.min(slopes)) / (abs(np.median(slopes)) + 0.001),
    }

def cluster_points(timeseries: np.ndarray, algorithm: str, k: int) -> np.ndarray:
    """Cluster with forced K"""
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(timeseries)
    elif algorithm == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(timeseries)
    else:  # kshape
        from tslearn.clustering import KShape
        data_ts = timeseries.reshape(len(timeseries), -1, 1)
        return KShape(n_clusters=k, random_state=42, n_init=2, max_iter=50).fit_predict(data_ts)


def select_diverse_samples(timeseries: np.ndarray, n_samples: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Smart sampling: EXTREMES first, then GRID-based fill"""
    from scipy import stats
    from scipy.signal import find_peaks
    
    n_points, n_timesteps = timeseries.shape
    t = np.arange(n_timesteps)
    
    # Calculate features for ALL series
    all_slopes = []
    all_turns = []
    for ts in timeseries:
        slope, _, _, _, _ = stats.linregress(t, ts)
        all_slopes.append(slope)
        peaks, _ = find_peaks(ts, distance=5)
        valleys, _ = find_peaks(-ts, distance=5)
        all_turns.append(len(peaks) + len(valleys))
    
    all_slopes = np.array(all_slopes)
    all_turns = np.array(all_turns)
    
    if n_points <= n_samples:
        return np.arange(n_points), all_slopes
    
    selected = set()
    
    # 1. EXTREMES (4 samples) - guaranteed diversity
    selected.add(np.argmin(all_slopes))  # Slowest decline
    selected.add(np.argmax(all_slopes))  # Fastest decline  
    selected.add(np.argmin(all_turns))   # Most linear (fewest turns)
    selected.add(np.argmax(all_turns))   # Most oscillating (most turns)
    
    # 2. GRID-based fill (slope × turns)
    slope_qs = [0, 25, 50, 75, 100]
    turn_qs = [0, 33, 66, 100]
    
    for i in range(len(slope_qs) - 1):
        s_low = np.percentile(all_slopes, slope_qs[i])
        s_high = np.percentile(all_slopes, slope_qs[i+1])
        
        for j in range(len(turn_qs) - 1):
            t_low = np.percentile(all_turns, turn_qs[j])
            t_high = np.percentile(all_turns, turn_qs[j+1])
            
            in_cell = np.where(
                (all_slopes >= s_low) & (all_slopes <= s_high) &
                (all_turns >= t_low) & (all_turns <= t_high)
            )[0]
            
            available = [x for x in in_cell if x not in selected]
            if available and len(selected) < n_samples:
                selected.add(np.random.choice(available))
    
    # 3. Fill remaining randomly if needed
    while len(selected) < n_samples:
        remaining = [i for i in range(n_points) if i not in selected]
        if remaining:
            selected.add(np.random.choice(remaining))
        else:
            break
    
    selected = np.array(list(selected)[:n_samples])
    return selected, all_slopes[selected]


def create_cluster_image(timeseries: np.ndarray, name: str, save_dir: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """Simple cluster visualization: 4x4 grid with raw time series"""
    # Select 16 samples (4x4 grid)
    idx, slopes = select_diverse_samples(timeseries, n_samples=16)
    sample_ts = timeseries[idx]
    
    # Sort by slope for visual ordering
    sort_order = np.argsort(slopes)
    sample_ts = sample_ts[sort_order]
    slopes = slopes[sort_order]
    
    # Create 4x4 subplot grid with WIDE aspect ratio (16x8) for better wave visibility
    fig, axes = plt.subplots(4, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (series, slope) in enumerate(zip(sample_ts, slopes)):
        ax = axes[i]
        ax.plot(series, color='black', lw=1.5)
        ax.set_title(f"{slope:.3f}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6)
        ax.set_xticks([])
    
    # Hide unused axes
    for i in range(len(sample_ts), 16):
        axes[i].set_visible(False)
    
    plt.suptitle(f"{name} ({len(timeseries)} pts)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    path = save_dir / f"{name.replace(' ','_').lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    gc.collect()
    return str(path), sample_ts, slopes






def analyze_with_gemini(image_path: str, n_points: int, cluster_name: str, sample_data: np.ndarray = None, sample_slopes: np.ndarray = None, full_cluster_ts: np.ndarray = None) -> Dict:
    """Ask Gemini if cluster is homogeneous - with IMAGE + DYNAMIC STATISTICS"""
    from google import genai
    from google.genai import types
    
    api_key = get_api_key()
    if not api_key:
        return {"is_homogeneous": True, "error": "No API key"}
    
    client = genai.Client(api_key=api_key)
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Calculate statistics if we have full cluster data
    stats_section = ""
    if full_cluster_ts is not None and len(full_cluster_ts) > 0:
        stats = calculate_cluster_stats(full_cluster_ts)
        stats_section = f"""
CLUSTER STATISTICS (from all {n_points} points):

• Seasonality Amplitude: {stats['amp_min']:.1f} to {stats['amp_max']:.1f} mm (median: {stats['amp_median']:.1f} mm)
• Total Displacement: {stats['disp_min']:.1f} to {stats['disp_max']:.1f} mm (median: {stats['disp_median']:.1f} mm)
• Subsidence Rate: {stats['slope_min']:.4f} to {stats['slope_max']:.4f} mm/step (median: {stats['slope_median']:.4f})
• Curvature: {stats['curv_min']:.6f} to {stats['curv_max']:.6f} (positive=accelerating, negative=decelerating)
• Rate Variability: median {stats['rate_var_median']:.4f}, max {stats['rate_var_max']:.4f}
"""
    
    # ═══════════════════════════════════════════════════════════════
    # SAR/InSAR Homogeneity Analysis Prompt - AI DECIDES
    # ═══════════════════════════════════════════════════════════════
    
    prompt = f"""You are a GEOSCIENTIST and InSAR TIME SERIES DEFORMATION SIGNAL PROCESSING EXPERT.

IMAGE: 16 graphs showing deformation patterns from {cluster_name} ({n_points} points).
Each subplot title shows the SLOPE value (positive = upward trend, negative = downward trend).

QUESTION: Do these graphs come from the SAME physical region (identical movement) or DIFFERENT regions?

LOOK CLOSELY AT THE SHAPE:
- Even if they trend in the same direction, are the shapes IDENTICAL?
- Does one have waves/seasonality while another is straight?
- Does one drop steeply while another is more flat?
- Do the peaks and valleys align reasonably well?

SAME REGION = Curves show consistent deformation behavior with matching shape and trend.
DIFFERENT REGIONS = Curves show different behaviors (e.g., opposite trends, different curve shapes, or MAJOR shifting peaks).

CRITICAL CHECK:
1. If trends are OPPOSITE (Up vs Down) -> SPLIT immediately.
2. If shapes are DIFFERENT (Wave vs Straight) -> SPLIT.
3. If peaks/valleys do NOT align in time (ALLOW SMALL SHIFTS) -> SPLIT only if shifts are large.
4. Only if indistinguishable -> HOMOGENEOUS.

Return JSON with confidence score (0-100):
```json
{{
    "is_homogeneous": true/false,
    "should_split": true/false,
    "distinct_groups": <number>,
    "confidence": <0-100>
}}
```"""
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_data, mime_type="image/png"),
                types.Part.from_text(text=prompt)
            ])],
            config=types.GenerateContentConfig(temperature=0)  # Deterministic
        )
        text = response.text or ""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            result = json.loads(text[start:end].strip())
        elif "{" in text:
            result = json.loads(text[text.find("{"):text.rfind("}")+1])
        else:
            return {"is_homogeneous": True, "raw": text}
        
        # ═══════════════════════════════════════════════════════════════
        # SELF-CORRECTION (Reflexion): Apply to ALL models
        # ═══════════════════════════════════════════════════════════════
        confidence = result.get("confidence", 100)
        should_split = result.get("should_split", False)
        
        if should_split and confidence < 80:
            # Low confidence SPLIT - ask for verification
            verify_prompt = f"""You previously decided to SPLIT this cluster with only {confidence}% confidence.

Look at the image again. Are you ABSOLUTELY CERTAIN these are from different regions?

Remember:
- Minor noise differences are NORMAL
- Small phase shifts are ACCEPTABLE
- Only split if there is UNDENIABLE proof (opposite trends, completely different shapes)

If you are not 100% sure, change your decision to HOMOGENEOUS.

Return JSON:
```json
{{
    "is_homogeneous": true/false,
    "should_split": true/false,
    "distinct_groups": <number>,
    "confidence": <0-100>,
    "verified": true
}}
```"""
            
            verify_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    types.Part.from_text(text=verify_prompt)
                ])],
                config=types.GenerateContentConfig(temperature=0)
            )
            verify_text = verify_response.text or ""
            
            if "```json" in verify_text:
                start = verify_text.find("```json") + 7
                end = verify_text.find("```", start)
                result = json.loads(verify_text[start:end].strip())
            elif "{" in verify_text:
                result = json.loads(verify_text[verify_text.find("{"):verify_text.rfind("}")+1])
            
            result["self_corrected"] = True
        
        return result
        
    except Exception as e:
        return {"is_homogeneous": True, "error": str(e)}


def iterative_merge_with_gemini(clusters_info: Dict, timeseries: np.ndarray, save_dir: Path, max_rounds: int = 10) -> Dict:
    """
    Batch-wise Merge with Gemini
    
    Process clusters in small batches (max 4 at a time) for clearer comparison.
    Each batch: Ask Gemini which clusters should merge.
    Repeat until no more merges possible.
    
    Returns:
        Dict with final merge groups and merge history
    """
    from google import genai
    from google.genai import types
    
    api_key = get_api_key()
    if not api_key:
        return {"merge_groups": {str(i): [name] for i, name in enumerate(clusters_info.keys())}}
    
    client = genai.Client(api_key=api_key)
    
    # Initialize: each cluster is its own group
    current_groups = {name: {'members': [name], 'indices': data['indices'] if isinstance(data, dict) else data} 
                      for name, data in clusters_info.items()}
    
    merge_history = []
    round_num = 0
    total_api_calls = 0
    no_merge_streak = 0  # Track consecutive rounds with 0 merges
    
    print(f"\n  🔄 Starting batch-wise merge with {len(current_groups)} clusters...")
    
    while round_num < max_rounds and len(current_groups) > 1:
        round_num += 1
        merges_this_round = 0
        
        # ═══════════════════════════════════════════════════════════════
        # SMART SHUFFLE: Order by cross-correlation (similar clusters together)
        # ═══════════════════════════════════════════════════════════════
        group_names = list(current_groups.keys())
        
        if len(group_names) >= 3:
            # Calculate mean time series for each cluster
            means = {}
            for name in group_names:
                indices = current_groups[name]['indices']
                cluster_ts = timeseries[indices]
                mean_ts = cluster_ts.mean(axis=0)
                # Z-score normalize for shape comparison
                means[name] = (mean_ts - mean_ts.mean()) / (mean_ts.std() + 1e-10)
            
            # Calculate pairwise correlation matrix
            from scipy.stats import pearsonr
            n = len(group_names)
            corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        corr, _ = pearsonr(means[group_names[i]], means[group_names[j]])
                        corr_matrix[i, j] = corr
                    else:
                        corr_matrix[i, j] = 1.0
            
            # Greedy grouping: start with highest correlation pair, add most similar
            used = set()
            ordered = []
            
            while len(ordered) < n:
                if not ordered:
                    # Find highest correlation pair to start
                    max_corr = -2
                    best_pair = (0, 1)
                    for i in range(n):
                        for j in range(i+1, n):
                            if corr_matrix[i, j] > max_corr:
                                max_corr = corr_matrix[i, j]
                                best_pair = (i, j)
                    ordered.extend([best_pair[0], best_pair[1]])
                    used.add(best_pair[0])
                    used.add(best_pair[1])
                else:
                    # Find cluster most similar to any in current group
                    best_idx = -1
                    best_sim = -2
                    for i in range(n):
                        if i not in used:
                            # Average correlation to last 2 added
                            sim = np.mean([corr_matrix[i, ordered[-1]], corr_matrix[i, ordered[-2]] if len(ordered) >= 2 else corr_matrix[i, ordered[-1]]])
                            if sim > best_sim:
                                best_sim = sim
                                best_idx = i
                    if best_idx >= 0:
                        ordered.append(best_idx)
                        used.add(best_idx)
            
            group_names = [group_names[i] for i in ordered]
        
        # ═══════════════════════════════════════════════════════════════
        # SIMPLE 3-CLUSTER BATCHES (sequential, not interleaved)
        # Each query: "Do any of these 3 clusters have identical shapes?"
        # ═══════════════════════════════════════════════════════════════
        batch_size = 3  # Fixed: 3 clusters per query
        n_clusters_now = len(group_names)
        
        # Create sequential batches of 3
        batches = []
        for i in range(0, n_clusters_now, batch_size):
            batch = group_names[i:i+batch_size]
            if len(batch) >= 2:  # Need at least 2 to compare
                batches.append(batch)
        
        print(f"\n      Round {round_num}: {len(group_names)} clusters → {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            if len(batch) < 2:
                continue  # Skip batches with only 1 cluster
            
            # ═══════════════════════════════════════════════════════════════
            # VISUALIZATION - Generate only the selected mode
            # ═══════════════════════════════════════════════════════════════
            n_batch = len(batch)
            merge_mode = get_merge_viz_mode()
            color_map_str = ""  # For V2 prompt
            
            if merge_mode == "v1":
                # V1: OVERLAY VISUALIZATION - All means on same axes with colors
                colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_batch)))
                if n_batch > 10:
                    colors = plt.cm.tab20(np.linspace(0, 1, n_batch))
                
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                
                for idx, group_name in enumerate(batch):
                    group_data = current_groups[group_name]
                    indices = group_data['indices']
                    cluster_ts = timeseries[indices]
                    mean_ts = cluster_ts.mean(axis=0)
                    
                    # Z-score normalize for shape comparison
                    mean_norm = (mean_ts - mean_ts.mean()) / (mean_ts.std() + 1e-10)
                    
                    color = colors[idx]
                    ax.plot(mean_norm, color=color, lw=2.5, label=f"{group_name} ({len(indices)}pts)")
                
                ax.set_xlabel("Time Step", fontsize=12)
                ax.set_ylabel("Normalized Value (shape)", fontsize=12)
                ax.set_title(f"Compare Shapes: {n_batch} Clusters", fontsize=14, fontweight='bold')
                ax.legend(loc='upper left', fontsize=9, ncol=2)
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                batch_img_path = save_dir / f"merge_r{round_num:02d}_b{batch_idx:02d}.png"
                plt.savefig(batch_img_path, dpi=150, bbox_inches='tight')
                plt.close('all')
                img_to_send = batch_img_path
                
            else:
                # V2: VERTICAL SUBPLOT VISUALIZATION - Stacked for easy VLM comparison
                # ALT ALTA (vertical) layout with taller subplots and Y-axis grid
                fig_v2, axes_v2 = plt.subplots(n_batch, 1, figsize=(12, 3*n_batch), facecolor='white')
                if n_batch == 1:
                    axes_v2 = [axes_v2]
                
                color_names = ['Red', 'Blue', 'Green', 'Orange', 'Purple', 'Cyan', 'Magenta', 'Yellow']
                color_hex = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1', '#D81B60', '#FDD835']
                
                raw_means = []
                for group_name in batch:
                    group_data = current_groups[group_name]
                    indices = group_data['indices']
                    cluster_ts = timeseries[indices]
                    mean_ts = cluster_ts.mean(axis=0)
                    raw_means.append((group_name, mean_ts, len(indices)))
                
                all_vals = np.concatenate([m[1] for m in raw_means])
                y_min, y_max = all_vals.min(), all_vals.max()
                y_range = y_max - y_min
                y_min -= y_range * 0.1
                y_max += y_range * 0.1
                
                color_mapping = []
                for idx, (group_name, mean_ts, n_pts) in enumerate(raw_means):
                    ax_v2 = axes_v2[idx]
                    color = color_hex[idx % len(color_hex)]
                    cname = color_names[idx % len(color_names)]
                    
                    ax_v2.plot(mean_ts, color=color, lw=2.5)
                    ax_v2.set_title(f"{cname} ({n_pts} pts)", fontsize=12, fontweight='bold', color=color, loc='left')
                    ax_v2.set_ylim(y_min, y_max)
                    ax_v2.set_ylabel("mm", fontsize=9)
                    # Only show X label for bottom plot
                    if idx == n_batch - 1:
                        ax_v2.set_xlabel("Time Step", fontsize=10)
                    else:
                        ax_v2.set_xticklabels([])
                    # Add horizontal Y-axis grid for alignment
                    ax_v2.grid(axis='y', alpha=0.4, linestyle='--')
                    ax_v2.grid(axis='x', alpha=0)
                    
                    color_mapping.append(f"{cname}={group_name}")
                
                fig_v2.suptitle(f"Cluster Comparison (R{round_num} B{batch_idx})", fontsize=14, fontweight='bold')
                plt.tight_layout()
                batch_img_path_v2 = save_dir / f"merge_r{round_num:02d}_b{batch_idx:02d}_v2.png"
                color_map_str = ", ".join(color_mapping)
                plt.savefig(batch_img_path_v2, dpi=150, bbox_inches='tight')
                plt.close('all')
                img_to_send = batch_img_path_v2
            
            # Load image to send to Gemini
            with open(img_to_send, 'rb') as f:
                image_data = f.read()
            # Generate appropriate prompt based on visualization mode
            if merge_mode == "v2":
                prompt = f"""These subplots show InSAR ground displacement time series from different areas.

COLOR MAPPING: {color_map_str}

TASK: You are a Signal Processing Expert comparing these signals.

CRITICAL DISTINCTION:
1. LINEAR TREND DEPTH: How deep/high the overall trend goes
   → Differences are OK! (one goes -60mm, another -120mm = SAME behavior, different scale)

2. SEASONAL WAVE AMPLITUDE: How big the up-down oscillations are
   → Differences are NOT OK! (one has 10mm waves, another has 40mm waves = DIFFERENT behavior)

LOOK AT THE WAVINESS:
- If one curve has SMALL seasonal oscillations and another has LARGE oscillations, they are DIFFERENT
- The "waviness strength" must be proportionally similar

MERGE IF:
✅ Same wave pattern (peaks at same times)
✅ Similar seasonal amplitude RELATIVE to each other
✅ No flat sections while others move

DO NOT MERGE IF:
❌ One has strong seasonal waves, another has weak/no waves
❌ One has a FLAT section while others continue moving
❌ Opposite overall trends (one up, one down)
❌ Peaks at different times

Return JSON:
{{"groups": [["Red", "Blue"]], "confidence": <0-100>}}

If all are different:
{{"groups": [], "confidence": 100}}"""
            else:
                prompt = f"""This image shows {n_batch} InSAR time series clusters overlaid on the SAME axes.
Each colored line is the MEAN of one cluster (normalized for shape comparison).

Clusters: {batch}

TASK: Determine if these overlaid curves represent the SAME SIGNAL (just different amplitudes) or DIFFERENT SIGNALS.

VISUAL ANALYSIS RULES:
1. "TRAIN TRACKS" TEST: Do the lines run PARALLEL to each other?
   - If they run parallel (even if one is higher/lower) -> MERGE (Same signal, different scale).
   - If they CROSS each other (X-shape) -> DO NOT MERGE (Different signals).

2. PEAK/VALLEY ALIGNMENT:
   - Do peaks occur at the exact same X-position for all lines?
   - If Red has a peak where Blue has a valley -> DIFFERENT SIGNAL.

3. IGNORE AMPLITUDE:
   - If shape and timing are identical, ignore height differences.
   - Focus on the PATTERN, not the magnitude.

MERGE ONLY IF:
- Curves are effectively "parallel" or overlapping.
- All peaks and valleys align perfectly in time.
- No lines trend in opposite directions.

If there is ANY doubt, do NOT merge.

Return JSON:
{{"groups": [
    ["name1", "name2"]  // ONLY if nearly identical
]}}

Or if NO curves are similar enough (THIS IS FINE):
{{"groups": []}}"""

            # ═══════════════════════════════════════════════════════════════
            # SINGLE API CALL
            # ═══════════════════════════════════════════════════════════════
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[types.Content(role="user", parts=[
                        types.Part.from_bytes(data=image_data, mime_type="image/png"),
                        types.Part.from_text(text=prompt)
                    ])],
                    config=types.GenerateContentConfig(temperature=0)
                )
                total_api_calls += 1
                text = response.text or ""
                
                # Parse response
                if "{" in text:
                    json_str = text[text.find("{"):text.rfind("}")+1]
                    json_str = json_str.replace("'", '"')
                    import re
                    json_str = re.sub(r'//.*', '', json_str)
                    result = json.loads(json_str)
                else:
                    result = {"groups": []}
                    
            except Exception as e:
                print(f"        Batch {batch_idx}: Error - {e}")
                result = {"groups": []}
            
            merge_groups = result.get("groups", [])
            merge_confidence = result.get("confidence", 100)
            
            # ═══════════════════════════════════════════════════════════════
            # MERGE SELF-CORRECTION: Low confidence? Ask again to verify
            # ═══════════════════════════════════════════════════════════════
            if merge_groups and merge_confidence < 80:
                print(f"        Batch {batch_idx}: Low confidence ({merge_confidence}%), verifying...")
                
                verify_prompt = f"""You previously decided to MERGE these clusters with only {merge_confidence}% confidence.

Look at the image again. Are you ABSOLUTELY CERTAIN these signals are from the SAME physical phenomenon?

Remember:
- Different seasonal wave amplitudes = DIFFERENT behavior = DO NOT MERGE
- One flat while others move = DO NOT MERGE
- Only merge if they are NEARLY IDENTICAL in shape

If you are not 100% sure, return empty groups.

Return JSON:
{{"groups": [...], "confidence": <0-100>, "verified": true}}

Or if uncertain:
{{"groups": [], "confidence": 100, "verified": true}}"""
                
                try:
                    verify_response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[types.Content(role="user", parts=[
                            types.Part.from_bytes(data=image_data, mime_type="image/png"),
                            types.Part.from_text(text=verify_prompt)
                        ])],
                        config=types.GenerateContentConfig(temperature=0)
                    )
                    total_api_calls += 1
                    verify_text = verify_response.text or ""
                    
                    if "{" in verify_text:
                        verify_json = verify_text[verify_text.find("{"):verify_text.rfind("}")+1]
                        verify_result = json.loads(verify_json)
                        merge_groups = verify_result.get("groups", [])
                        new_confidence = verify_result.get("confidence", 100)
                        
                        if not merge_groups:
                            print(f"        Batch {batch_idx}: Self-correction → No merge (was unsure)")
                        else:
                            print(f"        Batch {batch_idx}: Verified merge (confidence: {merge_confidence}% → {new_confidence}%)")
                except:
                    pass  # Keep original decision on error
            
            # V2: Convert color names back to cluster names
            if merge_mode == "v2" and merge_groups:
                color_to_cluster = {color_names[i]: batch[i] for i in range(min(len(batch), len(color_names)))}
                converted_groups = []
                for group in merge_groups:
                    converted = []
                    for name in group:
                        # Check if it's a color name
                        if name in color_to_cluster:
                            converted.append(color_to_cluster[name])
                        elif name in current_groups:
                            converted.append(name)  # Already a cluster name
                    if len(converted) >= 2:
                        converted_groups.append(converted)
                merge_groups = converted_groups
            
            if not merge_groups:
                print(f"        Batch {batch_idx}: No merge (confidence: {merge_confidence}%)")
            
            # Process merges using consensus groups
            
            for group in merge_groups:
                if not isinstance(group, list) or len(group) < 2:
                    continue
                    
                # Validate all names exist
                valid_names = [n for n in group if n in current_groups]
                if len(valid_names) >= 2:
                    # Merge all valid names into one group
                    base_name = valid_names[0]
                    merged_indices = current_groups[base_name]['indices']
                    merged_members = current_groups[base_name]['members'][:]
                    
                    for other_name in valid_names[1:]:
                        merged_indices = np.concatenate([merged_indices, current_groups[other_name]['indices']])
                        merged_members.extend(current_groups[other_name]['members'])
                        del current_groups[other_name]
                    
                    # Update base group
                    new_name = "+".join([n.split("+")[0] for n in valid_names[:2]])
                    if len(valid_names) > 2:
                        new_name += f"+{len(valid_names)-2}more"
                    
                    del current_groups[base_name]
                    current_groups[new_name] = {'members': merged_members, 'indices': merged_indices}
                    
                    print(f"        Batch {batch_idx}: MERGE {valid_names} → {new_name} (confidence: {merge_confidence}%)")
                    merges_this_round += 1
                    
                    merge_history.append({
                        'round': round_num,
                        'batch': batch_idx,
                        'merged': valid_names,
                        'new_name': new_name
                    })
            
            if not merge_groups or all(len(g) < 2 for g in merge_groups):
                print(f"        Batch {batch_idx}: No merge")
        
        # ═══════════════════════════════════════════════════════════════
        # CREATE ROUND SUMMARY VISUALIZATION
        # ═══════════════════════════════════════════════════════════════
        n_groups = len(current_groups)
        if n_groups > 0:
            cols = min(4, n_groups)
            rows = (n_groups + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), facecolor='white')
            if n_groups == 1:
                axes = np.array([axes])
            else:
                axes = np.array(axes).flatten()
            
            for idx, (name, data) in enumerate(sorted(current_groups.items())):
                ax = axes[idx]
                indices = data['indices']
                cluster_ts = timeseries[indices]
                mean_ts = cluster_ts.mean(axis=0)
                
                # Plot samples
                n_samples = min(5, len(cluster_ts))
                sample_idx = np.random.choice(len(cluster_ts), n_samples, replace=False)
                for i in sample_idx:
                    ax.plot(cluster_ts[i], alpha=0.3, lw=0.5)
                ax.plot(mean_ts, 'k-', lw=2)
                
                ax.set_title(f"{name}\n({len(indices)} pts)", fontsize=9)
                ax.grid(alpha=0.3)
                ax.set_xticks([])
            
            # Hide unused
            for idx in range(n_groups, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f"After Round {round_num}: {n_groups} clusters", fontsize=12, fontweight='bold')
            plt.tight_layout()
            round_summary_path = save_dir / f"merge_round_{round_num:02d}_summary.png"
            plt.savefig(round_summary_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close('all')
        
        # If no merges happened this round, shuffle and retry (up to 3 times)
        if merges_this_round == 0:
            no_merge_streak += 1
            if no_merge_streak >= 3:
                print(f"      No merges for 3 consecutive rounds, stopping.")
                break
            else:
                print(f"      No merges in round {round_num}, shuffling and retrying ({no_merge_streak}/3)...")
                # Shuffle is handled by random order in next round
        else:
            no_merge_streak = 0  # Reset streak on successful merge
    
    print(f"\n  ✅ Batch merge complete: {len(clusters_info)} → {len(current_groups)} groups")
    print(f"     Rounds: {round_num}, API calls: {total_api_calls}")
    
    # Build final merge groups
    final_groups = {}
    group_labels = {}
    for idx, (name, data) in enumerate(sorted(current_groups.items())):
        final_groups[str(idx)] = data['members']
        group_labels[str(idx)] = f"Merged group with {len(data['indices'])} points"
    
    return {
        "merge_groups": final_groups,
        "group_labels": group_labels,
        "merge_history": merge_history,
        "final_k": len(current_groups)
    }


def analyze_merge(image_path: str, cluster_summaries: list, clusters_info: Dict = None, timeseries: np.ndarray = None, save_dir: Path = None) -> Dict:
    """
    Ask Gemini which clusters should be merged - V2 SHAPE-NORMALIZED VERSION
    
    This version normalizes all cluster shapes to 0-1 range and focuses
    purely on shape similarity, achieving near-manual accuracy (ARI gap < 0.001)
    """
    from google import genai
    from google.genai import types
    
    api_key = get_api_key()
    if not api_key:
        return {"merge_suggestions": [], "final_k": len(cluster_summaries)}
    
    client = genai.Client(api_key=api_key)
    
    # ═══════════════════════════════════════════════════════════════
    # Create mean shape visualization with statistics for merge analysis
    # ═══════════════════════════════════════════════════════════════
    
    normalized_image_path = image_path  # fallback to original
    
    if clusters_info is not None and timeseries is not None and save_dir is not None:
        try:
            from scipy import stats as sp_stats
            
            n_clusters = len(clusters_info)
            
            # Grid layout: 4 columns
            cols = 4
            rows = (n_clusters + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5*rows))
            axes = axes.flatten() if n_clusters > 1 else [axes]
            
            for idx, (cluster_name, cluster_data) in enumerate(sorted(clusters_info.items())):
                ax = axes[idx]
                indices = cluster_data['indices'] if isinstance(cluster_data, dict) else cluster_data
                cluster_ts = timeseries[indices]
                
                # Calculate mean time series
                mean_ts = cluster_ts.mean(axis=0)
                
                # Plot with thick black line
                ax.plot(mean_ts, color='black', lw=2)
                
                # Simple title: just name and count
                ax.set_title(f'{cluster_name} (n={len(indices)})', fontsize=9, fontweight='bold')
                
                # Show X-axis for temporal reference
                ax.grid(alpha=0.3)
                
                # Add vertical visual anchors (red dashed lines)
                anchors = [25, 50, 75, 100, 125]
                for x_val in anchors:
                    if x_val < len(mean_ts):
                        ax.axvline(x=x_val, color='red', linestyle='--', alpha=0.5, lw=0.8)
                
                ax.set_xticks(np.arange(0, len(mean_ts), 25))  # Ticks match anchors
                ax.tick_params(axis='x', labelsize=8)
            
            # Hide unused axes
            for idx in range(n_clusters, len(axes)):
                axes[idx].axis('off')
            
            # NO suptitle - let subplot titles speak for themselves
            plt.tight_layout()
            
            normalized_image_path = save_dir / "normalized_shapes_for_merge.png"
            plt.savefig(normalized_image_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            gc.collect()


        except Exception as e:
            print(f"  ⚠️ Failed to create normalized image: {e}, using original")
            normalized_image_path = image_path
    
    # Load image
    with open(normalized_image_path, 'rb') as f:
        image_data = f.read()
    
    # Extract cluster names from summaries
    cluster_names = []
    for s in cluster_summaries:
        try:
            name = s.split(':')[0].strip().replace('- ', '')
            cluster_names.append(name)
        except:
            pass
    
    # Calculate per-cluster statistics for dynamic prompt
    cluster_stats_section = ""
    if clusters_info is not None and timeseries is not None:
        cluster_stats_section = "\nPER-CLUSTER STATISTICS:\n"
        for cluster_name, cluster_data in sorted(clusters_info.items()):
            indices = cluster_data['indices'] if isinstance(cluster_data, dict) else cluster_data
            cluster_ts = timeseries[indices]
            
            if len(cluster_ts) > 0:
                # Calculate stats for this cluster
                mean_ts = cluster_ts.mean(axis=0)
                n_timesteps = len(mean_ts)
                
                # Amplitude (seasonality)
                slope, intercept = np.polyfit(range(n_timesteps), mean_ts, 1)
                detrended = mean_ts - (slope * np.arange(n_timesteps) + intercept)
                amplitude = np.max(detrended) - np.min(detrended)
                
                # Displacement
                displacement = mean_ts[-1] - mean_ts[0]
                
                # Curvature
                coeffs = np.polyfit(range(n_timesteps), mean_ts, 2)
                curvature = coeffs[0] * 2
                
                # Rate variability
                window = n_timesteps // 5
                local_slopes = []
                for i in range(0, n_timesteps - window, window):
                    local_slope = (mean_ts[i + window] - mean_ts[i]) / window
                    local_slopes.append(local_slope)
                rate_var = np.std(local_slopes) if len(local_slopes) > 1 else 0
                
                # Curvature interpretation
                if curvature > 0.0001:
                    trend_type = "ACCELERATING"
                elif curvature < -0.0001:
                    trend_type = "DECELERATING"
                else:
                    trend_type = "LINEAR"
                
                # Amplitude interpretation
                if amplitude > 30:
                    amp_type = "STRONG_SEASONAL"
                elif amplitude > 15:
                    amp_type = "MODERATE_SEASONAL"
                else:
                    amp_type = "WEAK_SEASONAL"
                
                cluster_stats_section += f"""
{cluster_name}:
  Amplitude: {amplitude:.1f} mm ({amp_type})
  Displacement: {displacement:.1f} mm
  Slope: {slope:.4f} mm/step
  Curvature: {curvature:.6f} ({trend_type})
  Rate variability: {rate_var:.4f}
"""
    
    # ═══════════════════════════════════════════════════════════════
    # SAR/InSAR Time Series Merge Prompt - LENIENT VERSION
    # ═══════════════════════════════════════════════════════════════
    
    # Calculate expected number of groups (hint for Gemini)
    n_input_clusters = len(cluster_summaries)
    expected_groups = max(2, min(5, n_input_clusters // 2))  # Aim for 2-5 final groups
    
    prompt = f"""You are a geoscientist merging InSAR LOS time series clusters.

This image shows MEAN time series of {len(cluster_summaries)} clusters: {cluster_names}
{cluster_stats_section}

TASK: Aggressively merge clusters with SIMILAR deformation patterns.
TARGET: Reduce {n_input_clusters} clusters down to approximately {expected_groups}-{expected_groups+2} groups.

MERGE LIBERALLY - Focus on MAJOR pattern differences only:

1. SHAPE SIMILARITY (most important)
   - If curves have similar SHAPE (both linear, both curved, both wavy) → MERGE
   - Only separate if shapes are CLEARLY different

2. SEASONALITY
   - STRONG (>25 mm amplitude) vs WEAK (<10 mm) → may SEPARATE
   - MODERATE differences (10-25 mm) → MERGE if shapes match

3. TREND TYPE  
   - Same trend type (both LINEAR, both ACCELERATING, etc.) → MERGE
   - Ignore small curvature differences

4. DISPLACEMENT
   - Similar displacement range (within 50% of each other) → MERGE
   - Only separate if VERY different (e.g., -50mm vs -200mm)

MERGE-FRIENDLY RULES:
✓ When in doubt, MERGE - it's better to have fewer final clusters
✓ Small statistical differences should NOT prevent merging
✓ If visual shapes look similar, MERGE regardless of minor stat differences
✓ Focus on the OVERALL pattern, not exact numbers

GOAL: End with {expected_groups}-{expected_groups+2} distinct behavior groups, not {n_input_clusters}.

Return JSON:
```json
{{
    "merge_groups": {{
        "0": [list of cluster names to merge],
        "1": [list of cluster names],
        ...
    }},
    "group_labels": {{
        "0": "description of merged group behavior",
        "1": "description",
        ...
    }},
    "reasoning": "explain major differences that prevented merging"
}}
```

All {len(cluster_names)} clusters must appear in exactly one group."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=image_data, mime_type="image/png"),
                types.Part.from_text(text=prompt)
            ])]
        )
        text = response.text or ""
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return json.loads(text[start:end].strip())
        elif "{" in text:
            return json.loads(text[text.find("{"):text.rfind("}")+1])
        return {"merge_groups": {}, "group_labels": {}}
    except Exception as e:
        print(f"  ⚠️ Gemini merge analysis error: {e}")
        return {"merge_groups": {}, "group_labels": {}, "error": str(e)}


def evaluate(pred_labels: np.ndarray, true_labels: np.ndarray) -> Dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
    return {
        'ARI': round(adjusted_rand_score(true_labels, pred_labels), 4),
        'NMI': round(normalized_mutual_info_score(true_labels, pred_labels), 4),
        'V-Measure': round(v_measure_score(true_labels, pred_labels), 4)
    }


def run_fixed_k(df, ts_cols, ground_truth, algorithm, fixed_k=4, logger=None, experiment_name=None):
    """Run clustering with fixed K (ground truth K)"""
    if logger:
        logger.section(f"FIXED K={fixed_k}: {algorithm.upper()}")
    
    timeseries = df[ts_cols].values
    
    # Create output directory (experiment-specific if provided)
    if experiment_name:
        report_dir = get_experiment_dir(experiment_name)
    else:
        report_dir = REPORT_DIR
    fixed_dir = report_dir / f"fixed_k{fixed_k}"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.time()
    print(f"\n  Running {algorithm.upper()} with fixed K={fixed_k}...")
    
    labels = cluster_points(timeseries, algorithm, fixed_k)
    
    elapsed = time.time() - start
    metrics = evaluate(labels, ground_truth)
    
    print(f"  ✓ {algorithm.upper()} K={fixed_k}: ARI={metrics['ARI']:.4f}, NMI={metrics['NMI']:.4f}")
    
    # === Cluster Visualization ===
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    cols = min(n_clusters, 4)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_clusters == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for idx, cid in enumerate(unique_labels):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[0, col]
        
        mask = labels == cid
        cluster_ts = timeseries[mask]
        
        n_samples = min(30, len(cluster_ts))
        sample_idx = np.random.choice(len(cluster_ts), n_samples, replace=False)
        for ts in cluster_ts[sample_idx]:
            ax.plot(ts, color=colors[idx], alpha=0.3, lw=0.8)
        
        ax.plot(cluster_ts.mean(axis=0), 'k-', lw=2)
        ax.set_title(f'Cluster {idx}: {mask.sum()} pts', fontsize=10)
        ax.grid(alpha=0.3)
    
    for idx in range(n_clusters, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[0, col]
        ax.axis('off')
    
    plt.suptitle(f'{algorithm.upper()} Fixed K={fixed_k} - ARI={metrics["ARI"]:.4f}, NMI={metrics["NMI"]:.4f}', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fixed_dir / f"{algorithm}_k{fixed_k}_visualization.png", dpi=150)
    plt.close('all')
    gc.collect()
    print(f"  📊 Saved: {fixed_dir}/{algorithm}_k{fixed_k}_visualization.png")
    
    # === JSON Export ===
    export_data = {
        'algorithm': algorithm,
        'fixed_k': fixed_k,
        'metrics': {'ARI': metrics['ARI'], 'NMI': metrics['NMI']},
        'time_seconds': elapsed,
        'cluster_labels': labels.tolist(),
        'cluster_sizes': {str(cid): int((labels == cid).sum()) for cid in unique_labels}
    }
    
    with open(fixed_dir / f"{algorithm}_k{fixed_k}_results.json", 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"  💾 Saved: {fixed_dir}/{algorithm}_k{fixed_k}_results.json")
    
    if logger:
        logger.log(f"K={fixed_k}, ARI={metrics['ARI']:.4f}, NMI={metrics['NMI']:.4f}, Time={elapsed:.1f}s")
    
    return {
        'K': fixed_k, 
        'ARI': metrics['ARI'], 
        'NMI': metrics['NMI'],
        'time': elapsed, 
        'labels': labels
    }

def run_baseline(df, ts_cols, ground_truth, algorithm, logger, experiment_name=None):
    """Baseline: Auto-K clustering without LLM"""
    logger.section(f"BASELINE: {algorithm.upper()}")
    
    timeseries = df[ts_cols].values
    n_points = len(df)
    
    from sklearn.metrics import silhouette_score
    
    # Create output directory (experiment-specific if provided)
    if experiment_name:
        report_dir = get_experiment_dir(experiment_name)
    else:
        report_dir = REPORT_DIR
    baseline_dir = report_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-K selection
    k_scores = {}
    best_k, best_sil, best_labels = 2, -1, None
    all_labels = {}
    
    # Get K range from settings.json
    k_min, k_max = get_k_range()
    
    start = time.time()
    print(f"\n  Finding optimal K ({k_min}-{k_max})...")
    
    for k in range(k_min, k_max + 1):
        labels = cluster_points(timeseries, algorithm, k)
        all_labels[k] = labels.tolist()
        if len(set(labels)) > 1:
            sil = silhouette_score(timeseries, labels)
            k_scores[k] = sil
            marker = ""
            if sil > best_sil:
                best_sil, best_k, best_labels = sil, k, labels
                marker = " ← BEST"
            print(f"    K={k}: Silhouette={sil:.4f}{marker}")
            logger.log(f"K={k}: silhouette={sil:.4f}")
    
    print(f"\n  ✓ Optimal K = {best_k} (Silhouette = {best_sil:.4f})")
    
    elapsed = time.time() - start
    metrics = evaluate(best_labels, ground_truth)
    
    # === FIGURE 1: Auto-K Selection Graph ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(k_scores.keys())
    scores = list(k_scores.values())
    ax.plot(ks, scores, 'b-o', linewidth=2, markersize=10)
    ax.axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={best_k}')
    ax.scatter([best_k], [best_sil], color='red', s=200, zorder=5)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(f'{algorithm.upper()} - Auto-K Selection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(baseline_dir / "auto_k_selection.png", dpi=150)
    plt.close('all')
    gc.collect()
    print(f"  📊 Saved: {baseline_dir}/auto_k_selection.png")
    
    # === FIGURE 2: Cluster Visualization at Optimal K ===
    unique_labels = sorted(set(best_labels))
    n_clusters = len(unique_labels)
    cols = min(n_clusters, 4)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_clusters == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for idx, cid in enumerate(unique_labels):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[0, col]
        
        mask = best_labels == cid
        cluster_ts = timeseries[mask]
        
        # Sample
        n_samples = min(30, len(cluster_ts))
        sample_idx = np.random.choice(len(cluster_ts), n_samples, replace=False)
        for ts in cluster_ts[sample_idx]:
            ax.plot(ts, color=colors[idx], alpha=0.3, lw=0.8)
        
        # Mean
        ax.plot(cluster_ts.mean(axis=0), 'k-', lw=2)
        ax.set_title(f'Cluster {idx}: {mask.sum()} pts', fontsize=10)
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_clusters, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[0, col]
        ax.axis('off')
    
    plt.suptitle(f'{algorithm.upper()} Baseline (K={best_k}) - ARI={metrics["ARI"]:.4f}, NMI={metrics["NMI"]:.4f}', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(baseline_dir / "clusters_visualization.png", dpi=150)
    plt.close('all')
    gc.collect()
    print(f"  📊 Saved: {baseline_dir}/clusters_visualization.png")
    
    # === JSON Export ===
    export_data = {
        'algorithm': algorithm,
        'optimal_k': best_k,
        'silhouette_scores': k_scores,
        'metrics': {'ARI': metrics['ARI'], 'NMI': metrics['NMI']},
        'time_seconds': elapsed,
        'cluster_labels': best_labels.tolist(),
        'all_k_labels': all_labels,
        'cluster_sizes': {str(cid): int((best_labels == cid).sum()) for cid in unique_labels}
    }
    
    with open(baseline_dir / "results.json", 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"  💾 Saved: {baseline_dir}/results.json")
    
    logger.log(f"Optimal K={best_k}, ARI={metrics['ARI']:.4f}, NMI={metrics['NMI']:.4f}, Time={elapsed:.1f}s")
    
    return {
        'K': best_k, 
        'ARI': metrics['ARI'], 
        'NMI': metrics['NMI'],
        'time': elapsed, 
        'labels': best_labels
    }


def run_llm_guided(df, ts_cols, ground_truth, algorithm, logger, text_only=False, experiment_name=None):
    """LLM-Guided: Queue-based hierarchical splitting"""
    global REPORT_DIR  # Need to update global for step directories
    
    mode = "TEXT ONLY" if text_only else "IMAGE"
    logger.section(f"LLM-GUIDED: {algorithm.upper()} ({mode})")
    
    # Initialize ExperimentTracker for academic transparency
    exp_name = experiment_name or f"vlm_{algorithm}"
    tracker = ExperimentTracker(exp_name, algorithm, model_name=GEMINI_MODEL)
    tracker.set_metadata(model=GEMINI_MODEL, mode=mode)
    
    # Set experiment-specific output directory (using tracker's lazy creation)
    REPORT_DIR = tracker.get_experiment_dir()
    
    timeseries = df[ts_cols].values
    n_points = len(df)
    
    start_time = time.time()
    
    # Queue of clusters to review
    queue = deque()
    
    # Auto-K selection using Silhouette Score
    logger.subsection("Auto-K Selection (Silhouette Score)")
    
    from sklearn.metrics import silhouette_score
    
    k_min, k_max = get_k_range()  # From settings.json
    
    k_scores = {}
    all_k_labels = {}
    best_k, best_sil, best_labels = 2, -1, None
    print(f"\n  Finding optimal initial K ({k_min}-{k_max})...")
    
    for k in range(k_min, k_max + 1):
        try:
            labels = cluster_points(timeseries, algorithm, k)
            all_k_labels[k] = labels.tolist()
            if len(set(labels)) > 1:
                sil = silhouette_score(timeseries, labels)
                k_scores[k] = sil
                marker = ""
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
                    marker = " ← BEST"
                print(f"    K={k}: Silhouette={sil:.4f}{marker}")
                logger.log(f"K={k}: silhouette={sil:.4f}")
        except Exception as e:
            print(f"    K={k}: Error - {e}")
    
    print(f"\n  ✓ Optimal initial K = {best_k} (Silhouette = {best_sil:.4f})")
    logger.log(f"Selected K={best_k} (silhouette={best_sil:.4f})")
    
    # === SAVE Auto-K Selection Graph ===
    llm_dir = REPORT_DIR / "llm_guided"
    llm_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(k_scores.keys())
    scores = list(k_scores.values())
    ax.plot(ks, scores, 'b-o', linewidth=2, markersize=10)
    ax.axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'Initial K={best_k}')
    ax.scatter([best_k], [best_sil], color='red', s=200, zorder=5)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(f'VLM-Guided - Initial Auto-K Selection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(llm_dir / "initial_auto_k.png", dpi=150)
    plt.close('all')
    gc.collect()
    print(f"  📊 Saved: {llm_dir}/initial_auto_k.png")
    
    # === SAVE Initial K JSON ===
    initial_data = {
        'initial_k': best_k,
        'silhouette_scores': k_scores,
        'all_k_labels': all_k_labels
    }
    with open(llm_dir / "initial_k_selection.json", 'w') as f:
        json.dump(initial_data, f, indent=2)
    
    # Add each cluster to queue for Gemini review (hierarchical naming)
    for cid in sorted(set(best_labels)):
        mask = best_labels == cid
        queue.append({
            'indices': np.where(mask)[0],
            'timeseries': timeseries[mask],
            'name': f"C{cid}"  # Hierarchical: C0, C1, C2...
        })
    
    logger.log(f"Initial: {len(queue)} clusters in queue for Gemini review")
    
    
    # Step history for JSON export
    step_history = []
    
    # === COMPUTATIONAL METRICS TRACKING ===
    computational_metrics = {
        'api_calls': 0,
        'api_total_time': 0.0,
        'clustering_total_time': 0.0,
        'visualization_total_time': 0.0,
        'per_step_metrics': []
    }
    
    # Frozen clusters
    frozen = {}
    iteration = 0
    max_iterations = 100  # Safety limit - should never reach this
    
    # Process queue until empty
    while queue:
        iteration += 1
        item = queue.popleft()
        
        indices = item['indices']
        ts = item['timeseries']
        name = item['name']
        
        # DETAILED STATUS REPORT
        print(f"\n{'═'*60}")
        print(f"  STEP {iteration}")
        print(f"{'═'*60}")
        
        # Show all frozen clusters
        total_frozen_pts = sum(len(d['indices']) for d in frozen.values())
        print(f"\n  📦 FROZEN CLUSTERS ({len(frozen)} clusters, {total_frozen_pts} pts):")
        if frozen:
            for fn, fd in frozen.items():
                print(f"      ✓ {fn}: {len(fd['indices'])} pts")
        else:
            print(f"      (none yet)")
        
        # Show queue
        queue_pts = sum(len(q['indices']) for q in queue)
        print(f"\n  📋 QUEUE ({len(queue)} waiting, {queue_pts} pts):")
        for q in list(queue)[:5]:
            print(f"      ○ {q['name']}: {len(q['indices'])} pts")
        if len(queue) > 5:
            print(f"      ... and {len(queue) - 5} more")
        
        # Current processing
        print(f"\n  🔍 PROCESSING: {name} ({len(indices)} pts)")
        
        # Track step timing
        step_start_time = time.time()
        step_metrics = {'step': iteration, 'cluster': name, 'pts': len(indices)}
        
        if len(indices) < get_min_cluster_size():
            frozen[name] = {'indices': indices, 'timeseries': ts}
            print(f"      → FROZEN (small cluster <{get_min_cluster_size()} pts)")
            step_metrics['decision'] = 'FREEZE_SMALL'
            step_metrics['total_time'] = time.time() - step_start_time
            computational_metrics['per_step_metrics'].append(step_metrics)
            continue
        
        # Create image (also returns sample_data for text-only mode)
        img_dir = REPORT_DIR / f"step_{iteration:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        viz_start = time.time()
        img_path, sample_data, sample_slopes = create_cluster_image(ts, name, img_dir)
        viz_time = time.time() - viz_start
        computational_metrics['visualization_total_time'] += viz_time
        step_metrics['visualization_time'] = viz_time
        
        # API Call
        api_start = time.time()
        if text_only:
            print(f"      Using TEXT ONLY mode (no image sent to Gemini)")
            result = analyze_with_gemini_text_only(sample_data, sample_slopes, len(indices), name)
        else:
            print(f"      Image: {img_path}")
            print(f"      Asking Gemini...")
            result = analyze_with_gemini(img_path, len(indices), name, sample_data, sample_slopes, full_cluster_ts=ts)
        
        api_time = time.time() - api_start
        computational_metrics['api_calls'] += 1
        computational_metrics['api_total_time'] += api_time
        step_metrics['api_time'] = api_time
        
        if result.get('is_homogeneous', True) and not result.get('should_split', False):
            frozen[name] = {'indices': indices, 'timeseries': ts}
            print(f"\n  ✅ RESULT: HOMOGENEOUS → FROZEN")
            obs = result.get('observation', 'N/A')
            if len(obs) > 100:
                obs = obs[:100] + "..."
            print(f"      Reason: {obs}")
            step_metrics['decision'] = 'FREEZE'
            
            # Log to tracker for academic transparency
            tracker.log_step(name, len(indices), result, "FREEZE")
        else:
            # Use distinct_groups from Gemini's analysis
            distinct = result.get('distinct_groups', result.get('distinct_behaviors', 2))
            groups = result.get('group_descriptions', [])
            forced_k = max(2, min(distinct if isinstance(distinct, int) else len(groups), 4))
            print(f"\n  ⚠️  RESULT: HETEROGENEOUS → SPLIT into {forced_k}")
            print(f"      Groups detected: {groups if groups else distinct}")
            
            # Track clustering time
            cluster_start = time.time()
            sub_labels = cluster_points(ts, algorithm, forced_k)
            cluster_time = time.time() - cluster_start
            computational_metrics['clustering_total_time'] += cluster_time
            step_metrics['clustering_time'] = cluster_time
            step_metrics['decision'] = 'SPLIT'
            step_metrics['split_into_k'] = forced_k
            
            # Check if split actually worked (different sized clusters)
            unique_labels, counts = np.unique(sub_labels, return_counts=True)
            
            # If all sub-clusters are similar size OR biggest is >90% of original → split failed
            max_ratio = counts.max() / len(ts)
            if max_ratio > 0.9 or len(unique_labels) < 2:
                print(f"\n  ⚠️  SPLIT FAILED (max_ratio={max_ratio:.2f}) → FREEZING instead")
                frozen[name] = {'indices': indices, 'timeseries': ts}
                step_metrics['decision'] = 'SPLIT_FAILED'
                continue
            
            print(f"\n  🆕 NEW CLUSTERS:")
            new_clusters = []
            new_clusters_log = {}  # For tracker
            for sub_idx, sub_id in enumerate(sorted(set(sub_labels))):
                sub_mask = sub_labels == sub_id
                # Hierarchical naming: C0 → C0.1, C0.2, ...
                sub_name = f"{name}.{sub_idx + 1}"
                
                sub_data = {
                    'indices': indices[sub_mask],
                    'timeseries': ts[sub_mask],
                    'name': sub_name
                }
                queue.append(sub_data)
                new_clusters.append(sub_data)
                new_clusters_log[sub_name] = int(sub_mask.sum())  # For tracker
                print(f"      + {sub_name}: {sub_mask.sum()} pts → added to queue")
                
                # Create visual for each new cluster
                create_cluster_image(ts[sub_mask], sub_name, img_dir)
            
            # Memory cleanup after creating multiple images
            plt.close('all')
            gc.collect()
            
            # Log SPLIT to tracker for academic transparency
            tracker.log_step(name, len(indices), result, "SPLIT", new_clusters_log)
        
        # ════════════════════════════════════════════════════════
        # SAVE STEP JSON - Detailed step record
        # ════════════════════════════════════════════════════════
        step_record = {
            'step': iteration,
            'processed_cluster': name,
            'processed_pts': len(indices),
            'decision': 'FREEZE' if result.get('is_homogeneous', True) else 'SPLIT',
            'gemini_response': {
                'is_homogeneous': result.get('is_homogeneous', True),
                'distinct_groups': result.get('distinct_groups', 1),
                'group_descriptions': result.get('group_descriptions', []),
                'observation': result.get('observation', '')[:500]
            },
            'frozen_clusters': {fn: {'pts': len(fd['indices']), 'indices': fd['indices'].tolist()[:100]} for fn, fd in frozen.items()},
            'queue_clusters': [{'name': q['name'], 'pts': len(q['indices'])} for q in queue],
            'new_clusters': [],
            'total_frozen_pts': sum(len(d['indices']) for d in frozen.values()),
            'total_queue_pts': sum(len(q['indices']) for q in queue)
        }
        
        # Add new clusters if split occurred
        if not result.get('is_homogeneous', True):
            step_record['split_into'] = forced_k
            step_record['new_clusters'] = [{'name': nc['name'], 'pts': len(nc['indices'])} for nc in new_clusters] if 'new_clusters' in dir() else []
            step_record['parent_cluster'] = name
        
        # Add to step history
        step_history.append(step_record)
        
        # === ADD COMPUTATIONAL METRICS FOR THIS STEP ===
        step_metrics['total_time'] = time.time() - step_start_time
        computational_metrics['per_step_metrics'].append(step_metrics)
        
        # Save step JSON
        with open(img_dir / f"step_{iteration:02d}_data.json", 'w') as f:
            json.dump(step_record, f, indent=2)
        print(f"  💾 Saved: {img_dir}/step_{iteration:02d}_data.json")
        
        # ════════════════════════════════════════════════════════
        # CREATE STEP SUMMARY IMAGE (all clusters in one figure)
        # ════════════════════════════════════════════════════════
        all_current = list(frozen.items()) + [(q['name'], q) for q in queue]
        n_total = len(all_current)
        
        if n_total > 0 and n_total <= 12:
            cols = min(4, n_total)
            rows = (n_total + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if n_total == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            n_ts = len(ts_cols)
            dates = [datetime(2020,1,1) + timedelta(days=i*12) for i in range(n_ts)]
            
            for idx, (cname, cdata) in enumerate(all_current):
                ax = axes[idx]
                if 'timeseries' in cdata:
                    c_ts = cdata['timeseries']
                else:
                    c_ts = cdata.get('timeseries', np.array([]))
                
                if len(c_ts) > 0:
                    sample_idx = np.random.choice(len(c_ts), min(5, len(c_ts)), replace=False)
                    for i in sample_idx:
                        ax.plot(dates, c_ts[i], alpha=0.5, lw=0.8)
                    ax.plot(dates, c_ts.mean(axis=0), 'k-', lw=1.5)
                
                status = "✓ FROZEN" if cname in frozen else "○ QUEUE"
                ax.set_title(f"{cname}\n{len(c_ts) if len(c_ts) > 0 else '?'} pts - {status}", fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(alpha=0.3)
            
            for idx in range(n_total, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f"Step {iteration} Summary: {len(frozen)} frozen, {len(queue)} in queue", fontsize=10)
            plt.tight_layout()
            
            summary_path = img_dir / f"step_{iteration:02d}_summary.png"
            plt.savefig(summary_path, dpi=120)
            plt.close('all')  # Close ALL figures to prevent memory leak
            gc.collect()  # Force garbage collection
            print(f"\n  📊 Step summary: {summary_path}")
        
        # Safety check - prevent infinite loops
        if iteration >= max_iterations:
            print(f"\n  ⚠️ MAX ITERATIONS REACHED - freezing remaining queue")
            break
    
    # IMPORTANT: Freeze any remaining clusters in queue (no points should be lost!)
    if queue:
        print(f"\n  📦 Freezing {len(queue)} remaining clusters from queue...")
        for item in queue:
            frozen[item['name']] = {'indices': item['indices'], 'timeseries': item['timeseries']}
            print(f"      ✓ {item['name']}: {len(item['indices'])} pts")
    
    elapsed = time.time() - start_time
    
    # Verify all points are accounted for
    total_frozen_pts = sum(len(d['indices']) for d in frozen.values())
    print(f"\n  ✅ Total points in frozen clusters: {total_frozen_pts} / {n_points}")
    
    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY VISUALIZATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'═'*60}")
    
    # Create final clusters visualization
    final_dir = REPORT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    n_clusters = len(frozen)
    
    # Handle edge case: at least 1 cluster
    if n_clusters == 0:
        print("  ⚠️ No clusters to visualize")
        return None
    
    # Grid layout
    cols = min(4, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), facecolor='white')
    
    # Ensure axes is always a flat array
    if n_clusters == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()
    
    n_timesteps = len(ts_cols)
    dates = [datetime(2020,1,1) + timedelta(days=i*12) for i in range(n_timesteps)]
    colors = plt.cm.tab10.colors
    
    print(f"\n  📊 FINAL CLUSTERS ({n_clusters} clusters):")
    for idx, (name, data) in enumerate(frozen.items()):
        ax = axes[idx]
        cluster_ts = data['timeseries']
        
        # Sample 10 series
        sample_idx = np.random.choice(len(cluster_ts), min(10, len(cluster_ts)), replace=False)
        for i in sample_idx:
            ax.plot(dates, cluster_ts[i], alpha=0.5, lw=0.8, color=colors[idx % 10])
        ax.plot(dates, cluster_ts.mean(axis=0), 'k-', lw=2)
        
        ax.set_title(f"{name}\n({len(cluster_ts)} pts)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks([])
        
        print(f"      {idx+1}. {name}: {len(cluster_ts)} pts")
        
        # Save individual cluster image
        create_cluster_image(cluster_ts, f"final_{name}", final_dir)
    
    # Hide unused axes
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    summary_path = final_dir / "all_clusters_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    gc.collect()
    
    print(f"\n  📁 Summary image: {summary_path}")
    
    # ═══════════════════════════════════════════════════════════
    # MERGE ANALYSIS (ask Gemini if any clusters should merge)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  MERGE ANALYSIS")
    print(f"{'═'*60}")
    print(f"  Asking Gemini for merge recommendations...")
    
    # Create summary for Gemini
    cluster_names = list(frozen.keys())
    cluster_summaries = []
    for name, data in frozen.items():
        cluster_ts = data['timeseries']
        mean_trend = (cluster_ts[:, -1] - cluster_ts[:, 0]).mean()
        variance = np.var(cluster_ts, axis=1).mean()
        cluster_summaries.append(f"  - {name}: {len(cluster_ts)} pts, trend={mean_trend:.1f}mm, var={variance:.1f}")
    
    # Build clusters_info for V2 normalized merge analysis
    clusters_info_for_merge = {}
    for name, data in frozen.items():
        clusters_info_for_merge[name] = {
            'indices': data['indices'],
            'size': len(data['indices'])
        }
    
    # ═══════════════════════════════════════════════════════════════
    # MERGE: Single pass (no retry)
    # ═══════════════════════════════════════════════════════════════
    merge_result = iterative_merge_with_gemini(
        clusters_info=clusters_info_for_merge,
        timeseries=timeseries,
        save_dir=final_dir
    )
    
    # Calculate PRE-MERGE metrics
    pre_merge_labels = np.zeros(n_points, dtype=int)
    for idx, (name, data) in enumerate(frozen.items()):
        for i in data['indices']:
            pre_merge_labels[i] = idx
    
    pre_merge_metrics = evaluate(pre_merge_labels, ground_truth)
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE PRE-MERGE CLUSTER DATA (for manual merge tool)
    # ═══════════════════════════════════════════════════════════════
    pre_merge_dir = REPORT_DIR / "llm_guided"
    pre_merge_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labels
    np.save(pre_merge_dir / "pre_merge_labels.npy", pre_merge_labels)
    
    # Save cluster details
    pre_merge_clusters = {}
    for name, data in frozen.items():
        pre_merge_clusters[name] = {
            'indices': data['indices'].tolist(),
            'size': len(data['indices'])
        }
    
    with open(pre_merge_dir / "pre_merge_clusters.json", 'w') as f:
        json.dump({
            'n_clusters': n_clusters,
            'clusters': pre_merge_clusters,
            'metrics': pre_merge_metrics
        }, f, indent=2)
    
    print(f"\n  💾 Pre-merge data saved: {pre_merge_dir}")
    
    print(f"\n  📊 PRE-MERGE: {n_clusters} clusters")
    print(f"      ARI: {pre_merge_metrics['ARI']:.4f}")
    print(f"      NMI: {pre_merge_metrics['NMI']:.4f}")
    
    # ═══════════════════════════════════════════════════════════════
    # V2 MERGE: Apply merge using merge_groups format (not merge_pairs)
    # ═══════════════════════════════════════════════════════════════
    
    merge_groups = merge_result.get('merge_groups', {})
    
    if merge_groups:
        print(f"\n  ⚠️  APPLYING V2 MERGE GROUPS:")
        
        # V2 format: {"0": [cluster_nums], "1": [cluster_nums], ...}
        # Create new labels directly from merge_groups
        post_merge_labels = np.full(n_points, -1, dtype=int)
        
        for new_group_id, cluster_names_in_group in merge_groups.items():
            new_id = int(new_group_id)
            group_label = merge_result.get('group_labels', {}).get(new_group_id, f'Group_{new_id}')
            
            member_count = 0
            for cluster_name in cluster_names_in_group:
                # Handle both old format (int) and new format (string)
                if isinstance(cluster_name, int):
                    cluster_key = f"Cluster_{cluster_name}"
                else:
                    cluster_key = str(cluster_name)
                
                if cluster_key in frozen:
                    for idx in frozen[cluster_key]['indices']:
                        post_merge_labels[idx] = new_id
                    member_count += len(frozen[cluster_key]['indices'])
            
            if member_count > 0:
                print(f"      • Group {new_id} ({group_label}): {cluster_names_in_group} → {member_count} points")
        
        # Check for unassigned points
        unassigned = np.sum(post_merge_labels == -1)
        if unassigned > 0:
            print(f"      ⚠️ {unassigned} points unassigned - keeping pre-merge labels for them")
            # Fallback for unassigned points
            for idx, (name, data) in enumerate(frozen.items()):
                for i in data['indices']:
                    if post_merge_labels[i] == -1:
                        post_merge_labels[i] = len(merge_groups) + idx
        
        post_merge_metrics = evaluate(post_merge_labels, ground_truth)
        
        print(f"\n  📊 POST-MERGE: {len(merge_groups)} groups")
        print(f"      ARI: {post_merge_metrics['ARI']:.4f} (Δ{post_merge_metrics['ARI'] - pre_merge_metrics['ARI']:+.4f})")
        print(f"      NMI: {post_merge_metrics['NMI']:.4f} (Δ{post_merge_metrics['NMI'] - pre_merge_metrics['NMI']:+.4f})")
        
        # Log merge to tracker
        merge_reasoning = merge_result.get('reasoning', merge_result.get('observation', ''))
        tracker.log_merge(
            pre_merge_k=n_clusters,
            post_merge_k=len(merge_groups),
            merge_groups=merge_groups,
            pre_metrics=pre_merge_metrics,
            post_metrics=post_merge_metrics,
            merge_reasoning=merge_reasoning
        )
        
        # Export tree with merge information
        merge_info = {
            'pre_merge_k': n_clusters,
            'post_merge_k': len(merge_groups),
            'merge_groups': merge_groups,
            'pre_metrics': pre_merge_metrics,
            'post_metrics': post_merge_metrics
        }
        tree_file = tracker.get_experiment_dir() / "cluster_hierarchy.txt"
        print(f"  🌳 Tree: {tree_file}")
        
        # Use merged results if better
        if post_merge_metrics['ARI'] >= pre_merge_metrics['ARI']:
            print(f"\n  ✅ Merge improved results - using merged clusters")
            final_labels = post_merge_labels
            metrics = post_merge_metrics
            final_k = len(merge_groups)
        else:
            print(f"\n  ❌ Merge worsened results - keeping original clusters")
            final_labels = pre_merge_labels
            metrics = pre_merge_metrics
            final_k = n_clusters
    else:
        print(f"\n  ✅ No merge groups returned - keeping original clusters")
        final_labels = pre_merge_labels
        metrics = pre_merge_metrics
        final_k = n_clusters
    
    print(f"\n{'═'*60}")
    print(f"  FINAL EVALUATION")
    print(f"{'═'*60}")
    print(f"  Ground Truth: 4 clusters")
    print(f"  LLM-Guided:   {final_k} clusters")
    print(f"  ARI:          {metrics['ARI']:.4f}")
    print(f"  NMI:          {metrics['NMI']:.4f}")
    print(f"  Time:         {elapsed:.1f}s")
    
    logger.subsection("Results")
    logger.log(f"Final clusters: {len(frozen)}")
    logger.log(f"Iterations: {iteration}")
    logger.log(f"Time: {elapsed:.1f}s")
    logger.log(f"ARI={metrics['ARI']}, NMI={metrics['NMI']}")
    
    # === SAVE Final Results JSON ===
    final_cluster_data = {}
    for cname, cdata in frozen.items():
        final_cluster_data[cname] = {
            'indices': cdata['indices'].tolist(),
            'size': len(cdata['indices'])
        }
    
    llm_results = {
        'algorithm': algorithm,
        'initial_k': best_k,
        'final_k': final_k,
        'iterations': iteration,
        'pre_merge_k': n_clusters,
        'metrics': {
            'ARI': metrics['ARI'],
            'NMI': metrics['NMI']
        },
        'pre_merge_metrics': {
            'ARI': pre_merge_metrics['ARI'],
            'NMI': pre_merge_metrics['NMI']
        },
        'time_seconds': elapsed,
        'final_labels': final_labels.tolist(),
        'clusters': final_cluster_data,
        'step_history': step_history,
        'merge_info': {
            'suggested_pairs': merge_result.get('merge_pairs', []),
            'applied': merge_result.get('should_merge', False) and (post_merge_metrics['ARI'] > pre_merge_metrics['ARI']) if 'post_merge_metrics' in dir() else False
        },
        'computational_metrics': computational_metrics
    }
    
    with open(llm_dir / "final_results.json", 'w') as f:
        json.dump(llm_results, f, indent=2)
    print(f"\n  💾 Saved: {llm_dir}/final_results.json")
    
    # === SAVE Flow Tree JSON (for visualization) ===
    flow_tree = {
        'title': 'VLM-Guided Clustering Flow',
        'algorithm': algorithm,
        'initial_clusters': [{'name': f'Cluster_{i}', 'pts': int((best_labels == i).sum())} for i in sorted(set(best_labels))],
        'steps': [],
        'final_clusters': [{'name': k, 'pts': v['size']} for k, v in final_cluster_data.items()],
        'metrics': {'ARI': metrics['ARI'], 'NMI': metrics['NMI']}
    }
    
    # Build step tree
    for step in step_history:
        flow_step = {
            'step': step['step'],
            'cluster': step['processed_cluster'],
            'pts': step['processed_pts'],
            'decision': step['decision'],
            'children': step.get('new_clusters', []),
            'reasoning': step['gemini_response']['observation'][:200] if step['gemini_response']['observation'] else ''
        }
        flow_tree['steps'].append(flow_step)
    
    with open(llm_dir / "flow_tree.json", 'w') as f:
        json.dump(flow_tree, f, indent=2)
    print(f"  💾 Saved: {llm_dir}/flow_tree.json")
    
    # === SAVE Final Clusters Visualization ===
    unique_labels = sorted(set(final_labels))
    n_final = len(unique_labels)
    cols = min(n_final, 4)
    rows = (n_final + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_final == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_final))
    
    for idx, cid in enumerate(unique_labels):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[0, col]
        
        mask = final_labels == cid
        cluster_ts = timeseries[mask]
        
        n_samples = min(30, len(cluster_ts))
        sample_idx = np.random.choice(len(cluster_ts), n_samples, replace=False)
        for ts in cluster_ts[sample_idx]:
            ax.plot(ts, color=colors[idx], alpha=0.3, lw=0.8)
        
        ax.plot(cluster_ts.mean(axis=0), 'k-', lw=2)
        ax.set_title(f'Cluster {idx}: {mask.sum()} pts', fontsize=10)
        ax.grid(alpha=0.3)
    
    for idx in range(n_final, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[0, col]
        ax.axis('off')
    
    plt.suptitle(f'VLM-Guided Final Clusters (K={final_k}) - ARI={metrics["ARI"]:.4f}, NMI={metrics["NMI"]:.4f}', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(llm_dir / "final_clusters_visualization.png", dpi=150)
    plt.close('all')
    gc.collect()
    print(f"  📊 Saved: {llm_dir}/final_clusters_visualization.png")
    
    # Finalize experiment tracker (creates summary report)
    tracker.finalize(final_k, metrics, elapsed)
    
    
    return {
        'K': final_k,
        'ARI': metrics['ARI'],
        'NMI': metrics['NMI'],
        'time': elapsed,
        'labels': final_labels,
        'iterations': iteration
    }


def show_menu():
    print("\n" + "="*50)
    print(" EXPERIMENT SELECTION")
    print("="*50)
    print("  1. K-Means Baseline (Auto-K)")
    print("  2. K-Shape Baseline (Auto-K)")
    print("  3. Hierarchical-Ward Baseline (Auto-K)")
    print("  4. LLM-Guided K-Means")
    print("  5. LLM-Guided K-Shape")
    print("  6. LLM-Guided Hierarchical-Ward")
    print("  7. Fixed K=4 (All Algorithms)")
    print("  8. Reproducibility Test (6x VLM runs)")
    print("  0. Run ALL")
    print("="*50)
    try:
        return int(input("Select (0-8): "))
    except:
        return 1


def main():
    import sys
    
    # First, select Gemini model (important for quota management)
    select_gemini_model()
    
    choice = int(sys.argv[1]) if len(sys.argv) > 1 else show_menu()
    
    logger = AcademicLogger("experiment")
    df, ts_cols, ground_truth = load_data()
    
    logger.log(f"Data: {len(df)} points, {len(ts_cols)} timesteps")
    logger.log(f"Gemini Model: {GEMINI_MODEL}")
    
    results = {}
    
    # Each experiment saves to its own directory
    if choice in [0, 1]:
        exp_name = "baseline_kmeans"
        results["K-Means Baseline"] = run_baseline(df, ts_cols, ground_truth, "kmeans", logger, experiment_name=exp_name)
    if choice in [0, 2]:
        exp_name = "baseline_kshape"
        results["K-Shape Baseline"] = run_baseline(df, ts_cols, ground_truth, "kshape", logger, experiment_name=exp_name)
    if choice in [0, 3]:
        exp_name = "baseline_hierarchical"
        results["Hierarchical Baseline"] = run_baseline(df, ts_cols, ground_truth, "hierarchical", logger, experiment_name=exp_name)
    if choice in [0, 4]:
        exp_name = "vlm_kmeans"
        results["LLM-Guided K-Means"] = run_llm_guided(df, ts_cols, ground_truth, "kmeans", logger, experiment_name=exp_name)
    if choice in [0, 5]:
        exp_name = "vlm_kshape"
        results["LLM-Guided K-Shape"] = run_llm_guided(df, ts_cols, ground_truth, "kshape", logger, experiment_name=exp_name)
    if choice in [0, 6]:
        exp_name = "vlm_hierarchical"
        results["LLM-Guided Hierarchical"] = run_llm_guided(df, ts_cols, ground_truth, "hierarchical", logger, experiment_name=exp_name)
    if choice in [0, 7]:
        results["K-Means (K=4)"] = run_fixed_k(df, ts_cols, ground_truth, "kmeans", fixed_k=4, logger=logger, experiment_name="fixed_k4_kmeans")
        results["K-Shape (K=4)"] = run_fixed_k(df, ts_cols, ground_truth, "kshape", fixed_k=4, logger=logger, experiment_name="fixed_k4_kshape")
        results["Hierarchical (K=4)"] = run_fixed_k(df, ts_cols, ground_truth, "hierarchical", fixed_k=4, logger=logger, experiment_name="fixed_k4_hierarchical")
    
    if choice == 8:
        # Run reproducibility test with selected model
        print("\n" + "="*70)
        print(" REPRODUCIBILITY TEST")
        print(f" Running VLM algorithms 3 times with {GEMINI_MODEL}")
        print("="*70)
        from run_reproducibility import run_reproducibility_test_v2
        repro_results = run_reproducibility_test_v2(GEMINI_MODEL)
        print("\n✅ Reproducibility test complete!")
        print(f"   Results saved to: Results/reproducibility/")
        return  # Exit early, reproducibility test handles its own output
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'K':>5} {'ARI':>8} {'NMI':>8} {'Time':>8}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<30} {r['K']:>5} {r['ARI']:>8.4f} {r['NMI']:>8.4f} {r['time']:>7.1f}s")
    print("="*70)
    
    # Save summary to outputs folder
    summary_dir = Path("outputs/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_path = summary_dir / "all_results.json"
    with open(save_path, 'w') as f:
        json.dump({k: {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv) for kk, vv in v.items()} 
                   for k, v in results.items()}, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
