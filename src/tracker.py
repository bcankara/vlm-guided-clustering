"""
Experiment Tracker - Academic Transparency System
Logs every step of VLM-guided clustering for reproducibility
"""
import json
from pathlib import Path
from datetime import datetime

class ExperimentTracker:
    """Track VLM clustering experiments with detailed step-by-step logging"""
    
    def __init__(self, experiment_name: str, algorithm: str, base_dir: str = "Results", model_name: str = None):
        self.experiment_name = experiment_name
        self.algorithm = algorithm
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique experiment ID
        clean_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
        self.exp_id = f"{clean_name}_{self.timestamp}"
        
        # If model_name is provided, create subfolder for model
        if model_name:
            # Convert model name to folder-friendly format (e.g., gemini-2.5-pro -> Gemini_2-5_Pro)
            model_folder = model_name.replace("gemini-", "Gemini_").replace("-preview", "").replace(".", "-").replace("-", "_").replace("__", "_")
            self.exp_dir = self.base_dir / model_folder / self.exp_id
        else:
            self.exp_dir = self.base_dir / self.exp_id
        
        # Files (created lazily)
        self.log_file = self.exp_dir / "experiment_log.json"
        self.report_file = self.exp_dir / "experiment_report.md"
        
        # State
        self.steps = []
        self.metadata = {}
        self._created = False
        self.start_time = datetime.now()
    
    def _ensure_dir(self):
        """Create directory ONLY when first data is logged (Lazy Creation)"""
        if not self._created:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self._created = True
    
    def get_experiment_dir(self) -> Path:
        """Return experiment directory path (does NOT create it yet)"""
        return self.exp_dir
    
    def set_metadata(self, **kwargs):
        """Set experiment metadata (model, initial_k, etc.)"""
        self.metadata.update(kwargs)
    
    def log_step(self, cluster_name: str, n_points: int, gemini_response: dict, 
                 action: str, new_clusters: dict = None) -> Path:
        """
        Log a single VLM decision step
        
        Args:
            cluster_name: Name of cluster being processed (e.g., "Cluster_0")
            n_points: Number of points in cluster
            gemini_response: Raw response from Gemini
            action: "FREEZE" or "SPLIT"
            new_clusters: Dict of new cluster names and sizes if split
            
        Returns:
            Path to step visualization directory
        """
        self._ensure_dir()  # Create dirs on first log
        
        step_num = len(self.steps) + 1
        
        step_data = {
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "cluster": cluster_name,
            "n_points": int(n_points),
            "gemini_analysis": {
                "is_homogeneous": gemini_response.get("is_homogeneous", True),
                "distinct_groups": gemini_response.get("distinct_groups", 1),
                "reasoning": str(gemini_response.get("reasoning", gemini_response.get("observation", "")))[:300]
            },
            "action": action,
            "output": new_clusters if new_clusters else "FROZEN"
        }
        
        self.steps.append(step_data)
        
        # Update JSON log
        self._save_json_log()
        
        # Append to Markdown report
        self._append_to_report(step_data)
        
        # Return experiment directory (visuals are saved by main.py separately)
        return self.exp_dir
    
    def _save_json_log(self):
        """Save complete JSON log"""
        log_data = {
            "experiment": self.experiment_name,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "steps": self.steps
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def _append_to_report(self, step_data: dict):
        """Append step to Markdown report"""
        is_new = not self.report_file.exists()
        
        with open(self.report_file, 'a', encoding='utf-8') as f:
            if is_new:
                # Header
                f.write(f"# Experiment Report: {self.experiment_name}\n\n")
                f.write(f"**Algorithm:** {self.algorithm}  \n")
                f.write(f"**Model:** {self.metadata.get('model', 'N/A')}  \n")
                f.write(f"**Started:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
                f.write("---\n\n")
                f.write("## Processing Steps\n\n")
                f.write("| Step | Cluster | Points | Action | Result |\n")
                f.write("|:----:|---------|-------:|:------:|--------|\n")
            
            action_icon = "❄️" if step_data['action'] == "FREEZE" else "✂️"
            
            if isinstance(step_data['output'], dict):
                result_str = ", ".join([f"{k}({v})" for k, v in step_data['output'].items()])
            else:
                result_str = str(step_data['output'])
            
            row = f"| {step_data['step']} | **{step_data['cluster']}** | {step_data['n_points']:,} | {action_icon} | {result_str} |\n"
            f.write(row)
    
    def log_merge(self, pre_merge_k: int, post_merge_k: int, merge_groups: dict, 
                  pre_metrics: dict, post_metrics: dict, merge_reasoning: str = ""):
        """
        Log merge phase results
        
        Args:
            pre_merge_k: Number of clusters before merge
            post_merge_k: Number of clusters after merge
            merge_groups: Dict of merge groups {group_id: [cluster_nums]}
            pre_metrics: ARI/NMI before merge
            post_metrics: ARI/NMI after merge
            merge_reasoning: Gemini's reasoning for merge
        """
        self._ensure_dir()
        
        merge_data = {
            "phase": "MERGE",
            "timestamp": datetime.now().isoformat(),
            "pre_merge_clusters": pre_merge_k,
            "post_merge_clusters": post_merge_k,
            "merge_groups": merge_groups,
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "reasoning": merge_reasoning[:500] if merge_reasoning else ""
        }
        
        # Add to steps
        self.steps.append(merge_data)
        self._save_json_log()
        
        # Append to report
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write("\n---\n\n")
            f.write("## Merge Phase\n\n")
            f.write(f"- **Pre-merge Clusters:** {pre_merge_k}\n")
            f.write(f"- **Post-merge Clusters:** {post_merge_k}\n")
            f.write(f"- **Pre-merge ARI:** {pre_metrics.get('ARI', 0):.4f}\n")
            f.write(f"- **Post-merge ARI:** {post_metrics.get('ARI', 0):.4f}\n")
            
            if merge_groups:
                f.write("\n**Merge Groups:**\n")
                for group_id, clusters in merge_groups.items():
                    f.write(f"- Group {group_id}: {clusters}\n")
            
            if merge_reasoning:
                f.write(f"\n**Gemini Reasoning:** {merge_reasoning[:200]}...\n")
    
    def finalize(self, final_k: int, metrics: dict, elapsed_time: float):
        """Finalize experiment with summary"""
        self._ensure_dir()
        
        # Update metadata
        self.metadata['final_k'] = final_k
        self.metadata['metrics'] = metrics
        self.metadata['elapsed_seconds'] = elapsed_time
        self.metadata['total_steps'] = len(self.steps)
        
        # Save final JSON
        self._save_json_log()
        
        # Append summary to report
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write("\n---\n\n")
            f.write("## Final Results\n\n")
            f.write(f"- **Final Clusters:** {final_k}\n")
            f.write(f"- **ARI:** {metrics.get('ARI', 0):.4f}\n")
            f.write(f"- **NMI:** {metrics.get('NMI', 0):.4f}\n")
            f.write(f"- **Total Steps:** {len(self.steps)}\n")
            f.write(f"- **Duration:** {elapsed_time:.1f}s\n")
        
        print(f"  📝 Experiment log: {self.log_file}")
        print(f"  📄 Experiment report: {self.report_file}")
