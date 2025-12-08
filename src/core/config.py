"""
Keiba AI v3 - 設定
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "keiba.db"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
CSV_DIR = PROJECT_ROOT / "data" / "csv_imports"
OUTPUT_DIR = PROJECT_ROOT / "output"

LEARNING_CONFIG = {
    "min_sample_size": 100,
    "min_p_value": 0.05,
    "min_effect_size": 0.5,
    "min_validation_periods": 3,
}
