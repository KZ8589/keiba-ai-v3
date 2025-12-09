"""
予測ログ記録モジュール
predict_weekend.py の予測結果を prediction_logs テーブルに記録
"""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database import get_connection


class PredictionLogger:
    """予測ログを記録するクラス"""
    
    def __init__(self, model_version: str = "2.4"):
        """
        Args:
            model_version: モデルバージョン（デフォルト: 2.4）
        """
        self.model_version = model_version
        self.logged_count = 0
        self.skipped_count = 0
    
    def log_predictions(
        self,
        result_df: pd.DataFrame,
        features_df: pd.DataFrame = None,
        applied_patterns: dict = None,
        track_condition: str = "良",
        weather: str = "晴"
    ) -> dict:
        """
        予測結果を一括記録
        
        Args:
            result_df: predict_races()の戻り値DataFrame
            features_df: create_features()の戻り値DataFrame（オプション）
            applied_patterns: 適用したパターン辞書（オプション）
            track_condition: 馬場状態（デフォルト: 良）
            weather: 天気（デフォルト: 晴）
        
        Returns:
            dict: 記録結果 {logged: int, skipped: int, errors: list}
        """
        self.logged_count = 0
        self.skipped_count = 0
        errors = []
        
        print("="*60)
        print("📝 予測ログ記録開始")
        print("="*60)
        print(f"  対象レコード数: {len(result_df)}")
        print(f"  モデルバージョン: {self.model_version}")
        print(f"  馬場状態: {track_condition} / 天気: {weather}")
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            for idx, row in result_df.iterrows():
                try:
                    # 特徴量をJSON化
                    features_json = None
                    if features_df is not None and idx in features_df.index:
                        features_dict = features_df.loc[idx].to_dict()
                        # NumPy型をPython標準型に変換
                        features_dict = {
                            k: float(v) if pd.notna(v) else None 
                            for k, v in features_dict.items()
                        }
                        features_json = json.dumps(features_dict, ensure_ascii=False)
                    
                    # 適用パターンをJSON化
                    patterns_json = None
                    if applied_patterns:
                        patterns_json = json.dumps(applied_patterns, ensure_ascii=False)
                    
                    # INSERT OR REPLACE（重複時は更新）
                    cursor.execute("""
                        INSERT OR REPLACE INTO prediction_logs (
                            race_id, race_date, place_code, place_name, race_no,
                            track_type, distance, track_condition, weather, field_size,
                            horse_no, horse_name, horse_id,
                            pred_score, pred_rank,
                            features, applied_patterns,
                            odds_win, popularity,
                            model_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['race_id'],
                        row['date'],
                        row.get('place_code'),
                        row.get('place_name'),
                        int(row['race_no']) if pd.notna(row.get('race_no')) else None,
                        row.get('track_type'),
                        int(row['distance']) if pd.notna(row.get('distance')) else None,
                        track_condition,
                        weather,
                        int(row['field_size']) if pd.notna(row.get('field_size')) else None,
                        int(row['horse_no']),
                        row.get('horse_name'),
                        row.get('horse_id'),  # CSVにはないためNone
                        float(row['score']) if pd.notna(row.get('score')) else None,
                        int(row['pred_rank']) if pd.notna(row.get('pred_rank')) else None,
                        features_json,
                        patterns_json,
                        float(row['odds_win']) if pd.notna(row.get('odds_win')) else None,
                        int(row['popularity']) if pd.notna(row.get('popularity')) else None,
                        self.model_version,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    self.logged_count += 1
                    
                except Exception as e:
                    self.skipped_count += 1
                    errors.append({
                        'race_id': row.get('race_id'),
                        'horse_no': row.get('horse_no'),
                        'error': str(e)
                    })
            
            conn.commit()
        
        # 結果サマリー
        print()
        print("-"*60)
        print(f"✅ 記録完了: {self.logged_count}件")
        if self.skipped_count > 0:
            print(f"⚠️  スキップ: {self.skipped_count}件")
            for err in errors[:5]:  # 最初の5件のみ表示
                print(f"   - {err['race_id']} 馬番{err['horse_no']}: {err['error']}")
        print("="*60)
        
        return {
            'logged': self.logged_count,
            'skipped': self.skipped_count,
            'errors': errors
        }
    
    def log_single_prediction(
        self,
        race_id: str,
        race_date: str,
        horse_no: int,
        pred_score: float,
        pred_rank: int,
        **kwargs
    ) -> bool:
        """
        1頭分の予測を記録（デバッグ・テスト用）
        
        Args:
            race_id: レースID
            race_date: レース日付
            horse_no: 馬番
            pred_score: 予測スコア
            pred_rank: 予測順位
            **kwargs: その他のカラム値
        
        Returns:
            bool: 成功/失敗
        """
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                
                features_json = None
                if 'features' in kwargs and kwargs['features']:
                    features_json = json.dumps(kwargs['features'], ensure_ascii=False)
                
                patterns_json = None
                if 'applied_patterns' in kwargs and kwargs['applied_patterns']:
                    patterns_json = json.dumps(kwargs['applied_patterns'], ensure_ascii=False)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO prediction_logs (
                        race_id, race_date, place_code, place_name, race_no,
                        track_type, distance, track_condition, weather, field_size,
                        horse_no, horse_name, horse_id,
                        pred_score, pred_rank,
                        features, applied_patterns,
                        odds_win, popularity,
                        model_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    race_date,
                    kwargs.get('place_code'),
                    kwargs.get('place_name'),
                    kwargs.get('race_no'),
                    kwargs.get('track_type'),
                    kwargs.get('distance'),
                    kwargs.get('track_condition', '良'),
                    kwargs.get('weather', '晴'),
                    kwargs.get('field_size'),
                    horse_no,
                    kwargs.get('horse_name'),
                    kwargs.get('horse_id'),
                    pred_score,
                    pred_rank,
                    features_json,
                    patterns_json,
                    kwargs.get('odds_win'),
                    kwargs.get('popularity'),
                    self.model_version,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ ログ記録エラー: {e}")
            return False
    
    def get_logged_races(self, race_date: str = None) -> list:
        """
        記録済みのレースIDを取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            list: レースIDのリスト
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if race_date:
                cursor.execute(
                    "SELECT DISTINCT race_id FROM prediction_logs WHERE race_date = ?",
                    (race_date,)
                )
            else:
                cursor.execute("SELECT DISTINCT race_id FROM prediction_logs")
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_prediction_count(self, race_date: str = None) -> int:
        """
        記録済み予測数を取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            int: 予測数
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if race_date:
                cursor.execute(
                    "SELECT COUNT(*) FROM prediction_logs WHERE race_date = ?",
                    (race_date,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM prediction_logs")
            
            return cursor.fetchone()[0]


def test_logger():
    """テスト関数"""
    print("="*60)
    print("🧪 PredictionLogger テスト")
    print("="*60)
    
    logger = PredictionLogger(model_version="2.4-test")
    
    # テスト用DataFrame作成
    test_data = {
        'race_id': ['2025-12-09_05_01', '2025-12-09_05_01', '2025-12-09_05_01'],
        'date': ['2025-12-09', '2025-12-09', '2025-12-09'],
        'place_code': ['05', '05', '05'],
        'place_name': ['東京', '東京', '東京'],
        'race_no': [1, 1, 1],
        'track_type': ['芝', '芝', '芝'],
        'distance': [1600, 1600, 1600],
        'field_size': [16, 16, 16],
        'horse_no': [1, 2, 3],
        'horse_name': ['テスト馬A', 'テスト馬B', 'テスト馬C'],
        'score': [75.5, 68.2, 55.0],
        'pred_rank': [1, 2, 3],
        'odds_win': [3.5, 5.2, 12.0],
        'popularity': [2, 1, 5]
    }
    test_df = pd.DataFrame(test_data)
    
    # ログ記録テスト
    result = logger.log_predictions(test_df)
    
    print()
    print(f"📊 テスト結果:")
    print(f"  記録数: {result['logged']}")
    print(f"  スキップ: {result['skipped']}")
    
    # 確認
    count = logger.get_prediction_count('2025-12-09')
    races = logger.get_logged_races('2025-12-09')
    print(f"  DB内予測数: {count}")
    print(f"  DB内レース: {races}")
    
    # クリーンアップ（テストデータ削除）
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_logs WHERE model_version = '2.4-test'")
        conn.commit()
    print("  テストデータ削除完了")
    
    print()
    print("✅ テスト完了")


if __name__ == "__main__":
    test_logger()
