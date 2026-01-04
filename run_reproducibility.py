"""
Reproducibility Test for VLM-Guided Clustering
Runs each VLM algorithm multiple times to measure consistency

Author: Dr. Burak Can KARA
Affiliation: Amasya University
Email: burakcankara@gmail.com
Website: https://bcankara.com
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.settings import get_model, get_api_key

# Configuration
N_RUNS = 6
ALGORITHMS = ['kmeans', 'kshape', 'hierarchical']


def get_gemini_stamp(model_name: str) -> dict:
    """Get Gemini model info and timestamp for experiment tracking"""
    from google import genai
    from google.genai import types
    
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key", "model_requested": model_name}
    
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = """Lütfen sistem bilgilerini raporla:
1. Tam model adın ve versiyonun
2. Bilgi kesim tarihin (knowledge cutoff)
3. Şu anki tarih ve saat (UTC)

JSON formatında cevap ver:
```json
{
    "model_name": "model kimliğin",
    "model_version": "versiyon",
    "knowledge_cutoff": "bilgi kesim tarihi",
    "current_date": "YYYY-MM-DD",
    "current_time_utc": "HH:MM:SS"
}
```
Sadece JSON döndür."""

        response = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=prompt)
            ])]
        )
        text = response.text.strip()
        
        # Parse JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "model_requested": model_name}


def run_single_vlm_test(algorithm: str, run_number: int, model_name: str) -> dict:
    """Run a single VLM-guided clustering test"""
    import main
    
    # Ensure main uses our model
    main.GEMINI_MODEL = model_name
    
    # Load data
    df, ts_cols, ground_truth = main.load_data()
    logger = main.AcademicLogger(f"repro_{algorithm}_run{run_number}")
    
    # Create unique experiment name for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"vlm_{algorithm}_reproducibility_run{run_number}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"  {algorithm.upper()} - Run {run_number}/{N_RUNS}")
    print(f"  Model: {model_name}")
    print(f"={'='*60}")
    
    try:
        result = main.run_llm_guided(
            df, ts_cols, ground_truth, 
            algorithm, logger, 
            experiment_name=exp_name
        )
        
        return {
            'success': True,
            'algorithm': algorithm,
            'run': run_number,
            'model': model_name,
            'K': result['K'],
            'ARI': result['ARI'],
            'NMI': result['NMI'],
            'time': result['time'],
            'iterations': result.get('iterations', 0),
            'experiment_name': exp_name
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'algorithm': algorithm,
            'run': run_number,
            'model': model_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_reproducibility_test(model_name: str = None) -> dict:
    """
    Run reproducibility test for all VLM algorithms
    With checkpoint/resume: completed runs are not repeated
    
    Args:
        model_name: Gemini model to use (reads from settings.json if None)
    
    Returns:
        Dictionary with all results and statistics
    """
    # Get model from settings if not provided
    if model_name is None:
        model_name = get_model()
    
    # Progress file for checkpoint/resume
    progress_dir = Path("Results/reproducibility")
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / "progress.json"
    
    # Load existing progress
    completed_runs = {}  # {algo: [run_numbers]}
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_runs = progress_data.get('completed', {})
                print(f"\n  📋 Resuming from checkpoint: {sum(len(v) for v in completed_runs.values())} runs already done")
        except:
            pass
    
    print("\n" + "="*70)
    print("  VLM REPRODUCIBILITY TEST (with checkpoint/resume)")
    print("="*70)
    print(f"  Model: {model_name}")
    print(f"  Algorithms: {ALGORITHMS}")
    print(f"  Runs per algorithm: {N_RUNS}")
    print("="*70)
    
    # Get Gemini stamp
    print("\n  Querying Gemini for model info...")
    gemini_stamp = get_gemini_stamp(model_name)
    print(f"  Gemini response: {gemini_stamp}")
    
    # Results storage
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model_requested': model_name,
        'gemini_stamp': gemini_stamp,
        'algorithms': {},
        'summary': {}
    }
    
    # Run tests
    for algo in ALGORITHMS:
        print(f"\n{'#'*70}")
        print(f"  TESTING: {algo.upper()}")
        print(f"{'#'*70}")
        
        algo_results = []
        algo_completed = completed_runs.get(algo, [])
        
        for run in range(1, N_RUNS + 1):
            # Skip if already completed
            if run in algo_completed:
                print(f"\n  ⏭️  Run {run}: Already completed (skipping)")
                continue
            
            result = run_single_vlm_test(algo, run, model_name)
            algo_results.append(result)
            
            if result['success']:
                print(f"\n  ✅ Run {run}: K={result['K']}, ARI={result['ARI']:.4f}, NMI={result['NMI']:.4f}")
                
                # Save progress immediately
                if algo not in completed_runs:
                    completed_runs[algo] = []
                completed_runs[algo].append(run)
                
                with open(progress_file, 'w') as f:
                    json.dump({'completed': completed_runs, 'last_updated': datetime.now().isoformat()}, f, indent=2)
                print(f"      💾 Progress saved")
            else:
                print(f"\n  ❌ Run {run} Failed: {result['error']}")
        
        # Calculate statistics
        successful = [r for r in algo_results if r['success']]
        
        if successful:
            aris = [r['ARI'] for r in successful]
            nmis = [r['NMI'] for r in successful]
            ks = [r['K'] for r in successful]
            
            stats = {
                'n_successful': len(successful),
                'n_failed': len(algo_results) - len(successful),
                'ARI': {'mean': np.mean(aris), 'std': np.std(aris), 'min': min(aris), 'max': max(aris)},
                'NMI': {'mean': np.mean(nmis), 'std': np.std(nmis), 'min': min(nmis), 'max': max(nmis)},
                'K': {'values': ks, 'unique': list(set(ks)), 'consistent': len(set(ks)) == 1}
            }
        else:
            stats = {'n_successful': 0, 'n_failed': len(algo_results), 'error': 'All runs failed'}
        
        all_results['algorithms'][algo] = {
            'runs': algo_results,
            'statistics': stats
        }
        
        # Print summary for this algorithm
        if successful:
            print(f"\n  📊 {algo.upper()} Summary:")
            print(f"      ARI: {stats['ARI']['mean']:.4f} ± {stats['ARI']['std']:.4f}")
            print(f"      NMI: {stats['NMI']['mean']:.4f} ± {stats['NMI']['std']:.4f}")
            print(f"      K values: {stats['K']['values']} {'✓ Consistent' if stats['K']['consistent'] else '⚠️ Inconsistent'}")
    
    # Overall summary
    print("\n" + "="*70)
    print("  REPRODUCIBILITY SUMMARY")
    print("="*70)
    
    for algo, data in all_results['algorithms'].items():
        stats = data['statistics']
        if stats.get('n_successful', 0) > 0:
            print(f"\n  {algo.upper()}:")
            print(f"    Success Rate: {stats['n_successful']}/{stats['n_successful'] + stats['n_failed']}")
            print(f"    ARI: {stats['ARI']['mean']:.4f} ± {stats['ARI']['std']:.4f}")
            print(f"    NMI: {stats['NMI']['mean']:.4f} ± {stats['NMI']['std']:.4f}")
            print(f"    K Consistent: {'Yes' if stats['K']['consistent'] else 'No'}")
    
    # Save results
    output_dir = Path("Results/reproducibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reproducibility_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  💾 Results saved: {output_file}")
    
    # Clear progress file after successful completion
    if progress_file.exists():
        progress_file.unlink()
        print("  🧹 Progress cleared (all runs complete)")
    
    return all_results


def reset_progress():
    """Clear checkpoint progress to start fresh"""
    progress_file = Path("Results/reproducibility/progress.json")
    if progress_file.exists():
        progress_file.unlink()
        print("✅ Progress reset - will start from scratch")
    else:
        print("ℹ️ No progress file found")


# Alias for backward compatibility
run_reproducibility_test_v2 = run_reproducibility_test


if __name__ == "__main__":
    import sys
    
    if "--reset" in sys.argv:
        reset_progress()
    else:
        # When run directly, use model from settings.json
        results = run_reproducibility_test()
