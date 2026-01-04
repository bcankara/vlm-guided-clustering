"""
Centralized Settings Management
All configuration in one place, persisted to settings.json
"""
import json
from pathlib import Path
from datetime import datetime

SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

DEFAULT_SETTINGS = {
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-pro",
    "k_range": [2, 8],
    "max_iterations": 100,
    "min_cluster_size": 200,
    "merge_viz_mode": "v2",
    "last_updated": ""
}

def load_settings() -> dict:
    """Load settings from JSON file"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except json.JSONDecodeError:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict):
    """Save settings to JSON file"""
    settings["last_updated"] = datetime.now().isoformat()
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

def get_model() -> str:
    """Get current Gemini model"""
    return load_settings().get("gemini_model", "gemini-2.5-pro")

def set_model(model_name: str):
    """Set and save Gemini model"""
    settings = load_settings()
    settings["gemini_model"] = model_name
    save_settings(settings)

def get_api_key() -> str:
    """Get Gemini API key from settings.json"""
    return load_settings().get("gemini_api_key", "")

def set_api_key(api_key: str):
    """Set and save API key"""
    settings = load_settings()
    settings["gemini_api_key"] = api_key
    save_settings(settings)

def get_k_range() -> tuple:
    """Get K range for Auto-K selection"""
    settings = load_settings()
    k_range = settings.get("k_range", [2, 8])
    return (k_range[0], k_range[1])

def set_k_range(k_min: int, k_max: int):
    """Set K range"""
    settings = load_settings()
    settings["k_range"] = [k_min, k_max]
    save_settings(settings)

def get_min_cluster_size() -> int:
    """Get minimum cluster size for freezing small clusters"""
    return load_settings().get("min_cluster_size", 200)

def get_merge_viz_mode() -> str:
    """Get merge visualization mode (v1=overlay, v2=subplot)"""
    return load_settings().get("merge_viz_mode", "v1")

def set_merge_viz_mode(mode: str):
    """Set merge visualization mode"""
    if mode not in ["v1", "v2"]:
        raise ValueError("Mode must be 'v1' (overlay) or 'v2' (subplot)")
    settings = load_settings()
    settings["merge_viz_mode"] = mode
    save_settings(settings)

