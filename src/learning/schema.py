"""
自己学習システム - データベーススキーマ
"""
import sqlite3
from pathlib import Path

def create_learning_schema(db_path: str = None):
    """自己学習用テーブルを作成"""
    
    if db_path is None:
        db_path = Path(__file__).parent.parent.parent / "data" / "keiba.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*60)
    print("🗄️ 自己学習スキーマ構築")
    print("="*60)
    
    # 1. prediction_logs（予測ログ）
    cursor.execute("DROP TABLE IF EXISTS prediction_logs")
    cursor.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            race_date TEXT NOT NULL,
            place_code TEXT,
            place_name TEXT,
            race_no INTEGER,
            track_type TEXT,
            distance INTEGER,
            track_condition TEXT,
            weather TEXT,
            field_size INTEGER,
            horse_no INTEGER NOT NULL,
            horse_name TEXT,
            horse_id TEXT,
            pred_score REAL,
            pred_rank INTEGER,
            features JSON,
            applied_patterns JSON,
            odds_win REAL,
            popularity INTEGER,
            model_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_id, horse_no)
        )
    """)
    print("✅ 1. prediction_logs")
    
    # 2. prediction_results（予測結果）
    cursor.execute("DROP TABLE IF EXISTS prediction_results")
    cursor.execute("""
        CREATE TABLE prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL UNIQUE,
            race_date TEXT NOT NULL,
            pred_horse_no INTEGER,
            pred_horse_name TEXT,
            pred_score REAL,
            pred_odds REAL,
            actual_1st_no INTEGER,
            actual_1st_name TEXT,
            actual_1st_odds REAL,
            actual_1st_popularity INTEGER,
            actual_2nd_no INTEGER,
            actual_3rd_no INTEGER,
            is_hit_1st INTEGER,
            is_hit_top3 INTEGER,
            pred_1st_actual_pos INTEGER,
            upset_level TEXT,
            miss_reason_candidates JSON,
            analyzed_at TEXT
        )
    """)
    print("✅ 2. prediction_results")
    
    # 3. pattern_candidates（パターン候補）
    cursor.execute("DROP TABLE IF EXISTS pattern_candidates")
    cursor.execute("""
        CREATE TABLE pattern_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_name TEXT NOT NULL,
            pattern_conditions JSON NOT NULL,
            extracted_from TEXT,
            extraction_method TEXT,
            extraction_date TEXT,
            sample_size INTEGER,
            effect_size REAL,
            p_value REAL,
            validation_status TEXT DEFAULT 'pending',
            validation_result JSON,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✅ 3. pattern_candidates")
    
    # 4. validated_patterns（検証済みパターン）
    cursor.execute("DROP TABLE IF EXISTS validated_patterns")
    cursor.execute("""
        CREATE TABLE validated_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_name TEXT NOT NULL UNIQUE,
            pattern_description TEXT,
            pattern_conditions JSON NOT NULL,
            action_type TEXT,
            action_value REAL,
            action_description TEXT,
            sample_size INTEGER,
            win_rate REAL,
            baseline_rate REAL,
            effect_size REAL,
            p_value REAL,
            validation_method TEXT,
            validation_periods INTEGER,
            validation_consistency REAL,
            reasoning TEXT,
            evidence JSON,
            is_active INTEGER DEFAULT 1,
            activated_at TEXT,
            deactivated_at TEXT,
            deactivation_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✅ 4. validated_patterns")
    
    # 5. pattern_performance（パターン性能）
    cursor.execute("DROP TABLE IF EXISTS pattern_performance")
    cursor.execute("""
        CREATE TABLE pattern_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER NOT NULL,
            pattern_name TEXT,
            period_start TEXT,
            period_end TEXT,
            applied_count INTEGER,
            hit_count INTEGER,
            hit_rate REAL,
            expected_hit_rate REAL,
            lift REAL,
            calculated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pattern_id) REFERENCES validated_patterns(id)
        )
    """)
    print("✅ 5. pattern_performance")
    
    # 6. learning_history（学習履歴）
    cursor.execute("DROP TABLE IF EXISTS learning_history")
    cursor.execute("""
        CREATE TABLE learning_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id TEXT NOT NULL,
            cycle_type TEXT,
            analysis_period_start TEXT,
            analysis_period_end TEXT,
            races_analyzed INTEGER,
            predictions_analyzed INTEGER,
            candidates_extracted INTEGER,
            candidates_validated INTEGER,
            patterns_registered INTEGER,
            patterns_deactivated INTEGER,
            started_at TEXT,
            completed_at TEXT,
            status TEXT
        )
    """)
    print("✅ 6. learning_history")
    
    # インデックス
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_logs_race ON prediction_logs(race_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_logs_date ON prediction_logs(race_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_results_date ON prediction_results(race_date)")
    print("✅ インデックス作成")
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("✅ 自己学習スキーマ構築完了")
    print("="*60)


if __name__ == "__main__":
    create_learning_schema()
