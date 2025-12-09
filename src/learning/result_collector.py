"""
結果収集モジュール
race_results テーブルから実際の結果を取得し、prediction_results に記録
"""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database import get_connection


class ResultCollector:
    """レース結果を収集し、予測と比較するクラス"""
    
    def __init__(self):
        self.collected_count = 0
        self.skipped_count = 0
    
    def get_pending_races(self, race_date: str = None) -> list:
        """
        結果未収集のレースIDを取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            list: レースIDのリスト
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if race_date:
                # 指定日付の予測済み・結果未収集レース
                cursor.execute("""
                    SELECT DISTINCT pl.race_id
                    FROM prediction_logs pl
                    LEFT JOIN prediction_results pr ON pl.race_id = pr.race_id
                    WHERE pl.race_date = ? AND pr.race_id IS NULL
                """, (race_date,))
            else:
                # 全ての予測済み・結果未収集レース
                cursor.execute("""
                    SELECT DISTINCT pl.race_id
                    FROM prediction_logs pl
                    LEFT JOIN prediction_results pr ON pl.race_id = pr.race_id
                    WHERE pr.race_id IS NULL
                """)
            
            return [row[0] for row in cursor.fetchall()]
    
    def collect_results_from_db(self, race_ids: list) -> pd.DataFrame:
        """
        race_results テーブルから結果を取得
        
        Args:
            race_ids: レースIDのリスト
        
        Returns:
            DataFrame: 結果データ
        """
        if not race_ids:
            return pd.DataFrame()
        
        with get_connection() as conn:
            placeholders = ','.join(['?'] * len(race_ids))
            query = f"""
                SELECT 
                    race_id,
                    date,
                    horse_no,
                    horse_name,
                    finish_position,
                    odds_win,
                    popularity
                FROM race_results
                WHERE race_id IN ({placeholders})
                  AND finish_position IS NOT NULL
                  AND finish_position > 0
                ORDER BY race_id, finish_position
            """
            
            df = pd.read_sql_query(query, conn, params=race_ids)
        
        return df
    
    def get_predictions(self, race_ids: list) -> pd.DataFrame:
        """
        prediction_logs から予測データを取得
        
        Args:
            race_ids: レースIDのリスト
        
        Returns:
            DataFrame: 予測データ
        """
        if not race_ids:
            return pd.DataFrame()
        
        with get_connection() as conn:
            placeholders = ','.join(['?'] * len(race_ids))
            query = f"""
                SELECT 
                    race_id,
                    race_date,
                    horse_no,
                    horse_name,
                    pred_score,
                    pred_rank,
                    odds_win,
                    popularity
                FROM prediction_logs
                WHERE race_id IN ({placeholders})
                ORDER BY race_id, pred_rank
            """
            
            df = pd.read_sql_query(query, conn, params=race_ids)
        
        return df
    
    def compare_and_save(self, race_ids: list = None, race_date: str = None) -> dict:
        """
        予測と結果を比較し、prediction_results に保存
        
        Args:
            race_ids: レースIDのリスト（オプション）
            race_date: 日付フィルタ（オプション）
        
        Returns:
            dict: 処理結果 {collected: int, skipped: int, no_result: int, errors: list}
        """
        self.collected_count = 0
        self.skipped_count = 0
        no_result_count = 0
        errors = []
        
        print("="*60)
        print("📊 結果収集開始")
        print("="*60)
        
        # 対象レースを決定
        if race_ids is None:
            race_ids = self.get_pending_races(race_date)
        
        if not race_ids:
            print("  対象レースなし")
            return {
                'collected': 0,
                'skipped': 0,
                'no_result': 0,
                'errors': []
            }
        
        print(f"  対象レース数: {len(race_ids)}")
        
        # 予測データを取得
        pred_df = self.get_predictions(race_ids)
        if pred_df.empty:
            print("  予測データなし")
            return {
                'collected': 0,
                'skipped': 0,
                'no_result': 0,
                'errors': []
            }
        
        # 結果データを取得
        result_df = self.collect_results_from_db(race_ids)
        result_race_ids = set(result_df['race_id'].unique()) if not result_df.empty else set()
        
        print(f"  結果あり: {len(result_race_ids)}レース")
        print(f"  結果なし: {len(race_ids) - len(result_race_ids)}レース")
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            for race_id in race_ids:
                try:
                    # 予測1位の馬を取得
                    race_pred = pred_df[pred_df['race_id'] == race_id]
                    if race_pred.empty:
                        self.skipped_count += 1
                        continue
                    
                    pred_1st = race_pred[race_pred['pred_rank'] == 1].iloc[0]
                    
                    # 結果を取得
                    race_result = result_df[result_df['race_id'] == race_id]
                    
                    if race_result.empty:
                        # 結果がまだDBにない
                        no_result_count += 1
                        continue
                    
                    # 実際の1-3着を取得
                    actual_1st = race_result[race_result['finish_position'] == 1]
                    actual_2nd = race_result[race_result['finish_position'] == 2]
                    actual_3rd = race_result[race_result['finish_position'] == 3]
                    
                    if actual_1st.empty:
                        no_result_count += 1
                        continue
                    
                    actual_1st = actual_1st.iloc[0]
                    
                    # 予測1位馬の実際の着順を取得
                    pred_horse_result = race_result[
                        race_result['horse_no'] == pred_1st['horse_no']
                    ]
                    pred_1st_actual_pos = None
                    if not pred_horse_result.empty:
                        pred_1st_actual_pos = int(pred_horse_result.iloc[0]['finish_position'])
                    
                    # 的中判定
                    is_hit_1st = 1 if pred_1st['horse_no'] == actual_1st['horse_no'] else 0
                    is_hit_top3 = 1 if pred_1st_actual_pos and pred_1st_actual_pos <= 3 else 0
                    
                    # INSERT OR REPLACE
                    cursor.execute("""
                        INSERT OR REPLACE INTO prediction_results (
                            race_id, race_date,
                            pred_horse_no, pred_horse_name, pred_score, pred_odds,
                            actual_1st_no, actual_1st_name, actual_1st_odds, actual_1st_popularity,
                            actual_2nd_no, actual_3rd_no,
                            is_hit_1st, is_hit_top3, pred_1st_actual_pos,
                            analyzed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        race_id,
                        pred_1st['race_date'],
                        int(pred_1st['horse_no']),
                        pred_1st['horse_name'],
                        float(pred_1st['pred_score']) if pd.notna(pred_1st['pred_score']) else None,
                        float(pred_1st['odds_win']) if pd.notna(pred_1st['odds_win']) else None,
                        int(actual_1st['horse_no']),
                        actual_1st['horse_name'],
                        float(actual_1st['odds_win']) if pd.notna(actual_1st['odds_win']) else None,
                        int(actual_1st['popularity']) if pd.notna(actual_1st['popularity']) else None,
                        int(actual_2nd.iloc[0]['horse_no']) if not actual_2nd.empty else None,
                        int(actual_3rd.iloc[0]['horse_no']) if not actual_3rd.empty else None,
                        is_hit_1st,
                        is_hit_top3,
                        pred_1st_actual_pos,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    self.collected_count += 1
                    
                except Exception as e:
                    self.skipped_count += 1
                    errors.append({
                        'race_id': race_id,
                        'error': str(e)
                    })
            
            conn.commit()
        
        # 結果サマリー
        print()
        print("-"*60)
        print(f"✅ 収集完了: {self.collected_count}件")
        if no_result_count > 0:
            print(f"⏳ 結果待ち: {no_result_count}件")
        if self.skipped_count > 0:
            print(f"⚠️  スキップ: {self.skipped_count}件")
            for err in errors[:5]:
                print(f"   - {err['race_id']}: {err['error']}")
        print("="*60)
        
        return {
            'collected': self.collected_count,
            'skipped': self.skipped_count,
            'no_result': no_result_count,
            'errors': errors
        }
    
    def get_accuracy_summary(self, race_date: str = None) -> dict:
        """
        的中率サマリーを取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            dict: 的中率情報
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if race_date:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(is_hit_1st) as hit_1st,
                        SUM(is_hit_top3) as hit_top3
                    FROM prediction_results
                    WHERE race_date = ?
                """, (race_date,))
            else:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(is_hit_1st) as hit_1st,
                        SUM(is_hit_top3) as hit_top3
                    FROM prediction_results
                """)
            
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                total = row[0]
                hit_1st = row[1] or 0
                hit_top3 = row[2] or 0
                
                return {
                    'total': total,
                    'hit_1st': hit_1st,
                    'hit_top3': hit_top3,
                    'accuracy_1st': round(hit_1st / total * 100, 2),
                    'accuracy_top3': round(hit_top3 / total * 100, 2)
                }
            
            return {
                'total': 0,
                'hit_1st': 0,
                'hit_top3': 0,
                'accuracy_1st': 0.0,
                'accuracy_top3': 0.0
            }


def test_collector():
    """テスト関数"""
    print("="*60)
    print("🧪 ResultCollector テスト")
    print("="*60)
    
    # まず予測ログにテストデータを入れる
    from src.learning.prediction_logger import PredictionLogger
    
    logger = PredictionLogger(model_version="2.4-test")
    
    # race_resultsに存在するレースIDを使用（2024年のデータ）
    # まずDBから実際のレースIDを取得
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT race_id, date 
            FROM race_results 
            WHERE date >= '2024-01-01' 
            LIMIT 1
        """)
        row = cursor.fetchone()
    
    if not row:
        print("❌ テスト用レースデータがDBにありません")
        return
    
    test_race_id = row[0]
    test_date = row[1]
    print(f"  テスト用レースID: {test_race_id}")
    print(f"  テスト用日付: {test_date}")
    
    # 該当レースの馬情報を取得
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT horse_no, horse_name, odds_win, popularity, finish_position
            FROM race_results
            WHERE race_id = ?
            ORDER BY finish_position
            LIMIT 5
        """, (test_race_id,))
        horses = cursor.fetchall()
    
    if len(horses) < 3:
        print("❌ テスト用馬データが不足しています")
        return
    
    # テスト予測データ作成（2着馬を予測1位にする = 外れケース）
    test_data = []
    for i, h in enumerate(horses):
        test_data.append({
            'race_id': test_race_id,
            'date': test_date,
            'place_code': test_race_id.split('_')[1] if '_' in test_race_id else '05',
            'place_name': '東京',
            'race_no': 1,
            'track_type': '芝',
            'distance': 1600,
            'field_size': len(horses),
            'horse_no': h[0],
            'horse_name': h[1],
            'score': 90 - i * 10,
            'pred_rank': i + 1,
            'odds_win': h[2],
            'popularity': h[3]
        })
    
    # 予測1位を実際の2着馬にする（外れケースをテスト）
    if len(test_data) >= 2:
        test_data[0], test_data[1] = test_data[1], test_data[0]
        test_data[0]['pred_rank'] = 1
        test_data[0]['score'] = 90
        test_data[1]['pred_rank'] = 2
        test_data[1]['score'] = 80
    
    test_df = pd.DataFrame(test_data)
    
    # 予測ログを記録
    print()
    logger.log_predictions(test_df)
    
    # 結果収集テスト
    print()
    collector = ResultCollector()
    result = collector.compare_and_save(race_ids=[test_race_id])
    
    print()
    print(f"📊 テスト結果:")
    print(f"  収集数: {result['collected']}")
    print(f"  スキップ: {result['skipped']}")
    
    # 的中率確認
    accuracy = collector.get_accuracy_summary()
    print(f"  的中率(1着): {accuracy['accuracy_1st']}%")
    print(f"  的中率(Top3): {accuracy['accuracy_top3']}%")
    
    # prediction_results の内容確認
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT race_id, pred_horse_no, pred_horse_name, 
                   actual_1st_no, actual_1st_name,
                   is_hit_1st, is_hit_top3, pred_1st_actual_pos
            FROM prediction_results
            WHERE race_id = ?
        """, (test_race_id,))
        pr_row = cursor.fetchone()
    
    if pr_row:
        print()
        print("📋 prediction_results 内容:")
        print(f"  予測1位: {pr_row[1]}番 {pr_row[2]}")
        print(f"  実際1着: {pr_row[3]}番 {pr_row[4]}")
        print(f"  1着的中: {'✅' if pr_row[5] else '❌'}")
        print(f"  Top3的中: {'✅' if pr_row[6] else '❌'}")
        print(f"  予測1位の実際着順: {pr_row[7]}着")
    
    # クリーンアップ
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_logs WHERE model_version = '2.4-test'")
        cursor.execute("DELETE FROM prediction_results WHERE race_id = ?", (test_race_id,))
        conn.commit()
    print()
    print("  テストデータ削除完了")
    print()
    print("✅ テスト完了")


if __name__ == "__main__":
    test_collector()
