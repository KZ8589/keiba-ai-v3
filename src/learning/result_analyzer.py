"""
差分分析モジュール
予測と結果の差分を分析し、失敗要因候補を特定
"""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database import get_connection


class ResultAnalyzer:
    """予測結果を分析し、失敗要因を特定するクラス"""
    
    def __init__(self):
        self.analyzed_count = 0
        self.skipped_count = 0
    
    def get_unanalyzed_results(self, race_date: str = None) -> list:
        """
        未分析のレースIDを取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            list: レースIDのリスト
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if race_date:
                cursor.execute("""
                    SELECT race_id
                    FROM prediction_results
                    WHERE race_date = ? AND upset_level IS NULL
                """, (race_date,))
            else:
                cursor.execute("""
                    SELECT race_id
                    FROM prediction_results
                    WHERE upset_level IS NULL
                """)
            
            return [row[0] for row in cursor.fetchall()]
    
    def determine_upset_level(
        self,
        is_hit_1st: int,
        pred_1st_actual_pos: int,
        actual_1st_popularity: int,
        actual_1st_odds: float
    ) -> str:
        """
        波乱度を判定
        
        Args:
            is_hit_1st: 1着的中フラグ
            pred_1st_actual_pos: 予測1位の実際着順
            actual_1st_popularity: 実際1着の人気
            actual_1st_odds: 実際1着のオッズ
        
        Returns:
            str: 波乱度 (none/minor/major/extreme)
        """
        # 的中の場合
        if is_hit_1st == 1:
            return 'none'
        
        # 予測1位の着順による判定
        if pred_1st_actual_pos is None:
            return 'extreme'  # 着順不明（出走取消など）
        
        if pred_1st_actual_pos <= 3:
            return 'minor'  # 2-3着
        elif pred_1st_actual_pos <= 6:
            return 'major'  # 4-6着
        else:
            return 'extreme'  # 7着以下
    
    def generate_miss_reasons(
        self,
        race_id: str,
        pred_data: dict,
        result_data: dict,
        race_context: dict
    ) -> list:
        """
        失敗要因候補を生成
        
        Args:
            race_id: レースID
            pred_data: 予測データ
            result_data: 結果データ
            race_context: レースコンテキスト（馬場、天気など）
        
        Returns:
            list: 失敗要因候補のリスト
        """
        reasons = []
        
        # 1. 人気薄の激走
        actual_pop = result_data.get('actual_1st_popularity')
        actual_odds = result_data.get('actual_1st_odds')
        
        if actual_pop and actual_pop >= 6:
            reasons.append({
                'type': 'upset_winner',
                'description': f'人気薄（{actual_pop}番人気）が勝利',
                'severity': 'high' if actual_pop >= 10 else 'medium',
                'details': {
                    'winner_popularity': actual_pop,
                    'winner_odds': actual_odds
                }
            })
        
        # 2. 高オッズ馬の激走
        if actual_odds and actual_odds >= 20.0:
            reasons.append({
                'type': 'high_odds_winner',
                'description': f'高オッズ馬（{actual_odds:.1f}倍）が勝利',
                'severity': 'high' if actual_odds >= 50.0 else 'medium',
                'details': {
                    'winner_odds': actual_odds
                }
            })
        
        # 3. 予測スコアと結果の乖離
        pred_score = pred_data.get('pred_score')
        pred_1st_actual_pos = result_data.get('pred_1st_actual_pos')
        
        if pred_score and pred_score >= 70 and pred_1st_actual_pos and pred_1st_actual_pos >= 5:
            reasons.append({
                'type': 'high_confidence_miss',
                'description': f'高スコア（{pred_score:.1f}%）が{pred_1st_actual_pos}着に惨敗',
                'severity': 'high',
                'details': {
                    'pred_score': pred_score,
                    'actual_position': pred_1st_actual_pos
                }
            })
        
        # 4. 馬場状態の影響（コンテキストがある場合）
        track_condition = race_context.get('track_condition')
        if track_condition and track_condition in ['重', '不良']:
            reasons.append({
                'type': 'track_condition',
                'description': f'馬場状態（{track_condition}）の影響可能性',
                'severity': 'medium',
                'details': {
                    'track_condition': track_condition
                }
            })
        
        # 5. 天気の影響
        weather = race_context.get('weather')
        if weather and weather in ['雨', '雪']:
            reasons.append({
                'type': 'weather',
                'description': f'天気（{weather}）の影響可能性',
                'severity': 'low',
                'details': {
                    'weather': weather
                }
            })
        
        # 6. 予測オッズと実際オッズの乖離
        pred_odds = pred_data.get('pred_odds')
        if pred_odds and actual_odds:
            odds_ratio = actual_odds / pred_odds if pred_odds > 0 else 0
            if odds_ratio >= 3.0:
                reasons.append({
                    'type': 'odds_gap',
                    'description': f'予測本命（{pred_odds:.1f}倍）と勝馬（{actual_odds:.1f}倍）のオッズ乖離',
                    'severity': 'medium',
                    'details': {
                        'pred_odds': pred_odds,
                        'winner_odds': actual_odds,
                        'odds_ratio': round(odds_ratio, 2)
                    }
                })
        
        # 7. 距離の影響（将来拡張用）
        distance = race_context.get('distance')
        track_type = race_context.get('track_type')
        if distance and distance >= 2400:
            reasons.append({
                'type': 'long_distance',
                'description': f'長距離レース（{distance}m）でのスタミナ要因',
                'severity': 'low',
                'details': {
                    'distance': distance,
                    'track_type': track_type
                }
            })
        
        # 要因がない場合
        if not reasons:
            reasons.append({
                'type': 'unknown',
                'description': '明確な要因特定困難',
                'severity': 'low',
                'details': {}
            })
        
        return reasons
    
    def analyze_results(self, race_ids: list = None, race_date: str = None) -> dict:
        """
        予測結果を分析し、失敗要因を特定
        
        Args:
            race_ids: レースIDのリスト（オプション）
            race_date: 日付フィルタ（オプション）
        
        Returns:
            dict: 分析結果 {analyzed: int, skipped: int, errors: list}
        """
        self.analyzed_count = 0
        self.skipped_count = 0
        errors = []
        
        print("="*60)
        print("🔍 差分分析開始")
        print("="*60)
        
        # 対象レースを決定
        if race_ids is None:
            race_ids = self.get_unanalyzed_results(race_date)
        
        if not race_ids:
            print("  対象レースなし")
            return {
                'analyzed': 0,
                'skipped': 0,
                'errors': []
            }
        
        print(f"  対象レース数: {len(race_ids)}")
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            for race_id in race_ids:
                try:
                    # prediction_results からデータ取得
                    cursor.execute("""
                        SELECT 
                            race_id, race_date,
                            pred_horse_no, pred_horse_name, pred_score, pred_odds,
                            actual_1st_no, actual_1st_name, actual_1st_odds, actual_1st_popularity,
                            is_hit_1st, is_hit_top3, pred_1st_actual_pos
                        FROM prediction_results
                        WHERE race_id = ?
                    """, (race_id,))
                    
                    pr_row = cursor.fetchone()
                    if not pr_row:
                        self.skipped_count += 1
                        continue
                    
                    # prediction_logs からコンテキスト取得
                    cursor.execute("""
                        SELECT track_type, distance, track_condition, weather
                        FROM prediction_logs
                        WHERE race_id = ?
                        LIMIT 1
                    """, (race_id,))
                    
                    pl_row = cursor.fetchone()
                    
                    # データ構造化
                    pred_data = {
                        'pred_horse_no': pr_row[2],
                        'pred_horse_name': pr_row[3],
                        'pred_score': pr_row[4],
                        'pred_odds': pr_row[5]
                    }
                    
                    result_data = {
                        'actual_1st_no': pr_row[6],
                        'actual_1st_name': pr_row[7],
                        'actual_1st_odds': pr_row[8],
                        'actual_1st_popularity': pr_row[9],
                        'is_hit_1st': pr_row[10],
                        'is_hit_top3': pr_row[11],
                        'pred_1st_actual_pos': pr_row[12]
                    }
                    
                    race_context = {}
                    if pl_row:
                        race_context = {
                            'track_type': pl_row[0],
                            'distance': pl_row[1],
                            'track_condition': pl_row[2],
                            'weather': pl_row[3]
                        }
                    
                    # 波乱度判定
                    upset_level = self.determine_upset_level(
                        is_hit_1st=result_data['is_hit_1st'],
                        pred_1st_actual_pos=result_data['pred_1st_actual_pos'],
                        actual_1st_popularity=result_data['actual_1st_popularity'],
                        actual_1st_odds=result_data['actual_1st_odds']
                    )
                    
                    # 失敗要因生成（的中以外の場合）
                    miss_reasons = []
                    if upset_level != 'none':
                        miss_reasons = self.generate_miss_reasons(
                            race_id=race_id,
                            pred_data=pred_data,
                            result_data=result_data,
                            race_context=race_context
                        )
                    
                    # prediction_results を更新
                    cursor.execute("""
                        UPDATE prediction_results
                        SET upset_level = ?,
                            miss_reason_candidates = ?,
                            analyzed_at = ?
                        WHERE race_id = ?
                    """, (
                        upset_level,
                        json.dumps(miss_reasons, ensure_ascii=False) if miss_reasons else None,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        race_id
                    ))
                    
                    self.analyzed_count += 1
                    
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
        print(f"✅ 分析完了: {self.analyzed_count}件")
        if self.skipped_count > 0:
            print(f"⚠️  スキップ: {self.skipped_count}件")
            for err in errors[:5]:
                print(f"   - {err['race_id']}: {err['error']}")
        print("="*60)
        
        return {
            'analyzed': self.analyzed_count,
            'skipped': self.skipped_count,
            'errors': errors
        }
    
    def get_analysis_summary(self, race_date: str = None) -> dict:
        """
        分析サマリーを取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            dict: 分析サマリー
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            base_query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN upset_level = 'none' THEN 1 ELSE 0 END) as none_count,
                    SUM(CASE WHEN upset_level = 'minor' THEN 1 ELSE 0 END) as minor_count,
                    SUM(CASE WHEN upset_level = 'major' THEN 1 ELSE 0 END) as major_count,
                    SUM(CASE WHEN upset_level = 'extreme' THEN 1 ELSE 0 END) as extreme_count
                FROM prediction_results
            """
            
            if race_date:
                cursor.execute(base_query + " WHERE race_date = ?", (race_date,))
            else:
                cursor.execute(base_query)
            
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                total = row[0]
                return {
                    'total': total,
                    'none': row[1] or 0,
                    'minor': row[2] or 0,
                    'major': row[3] or 0,
                    'extreme': row[4] or 0,
                    'none_pct': round((row[1] or 0) / total * 100, 2),
                    'minor_pct': round((row[2] or 0) / total * 100, 2),
                    'major_pct': round((row[3] or 0) / total * 100, 2),
                    'extreme_pct': round((row[4] or 0) / total * 100, 2)
                }
            
            return {
                'total': 0,
                'none': 0, 'minor': 0, 'major': 0, 'extreme': 0,
                'none_pct': 0.0, 'minor_pct': 0.0, 'major_pct': 0.0, 'extreme_pct': 0.0
            }
    
    def get_miss_reason_stats(self, race_date: str = None) -> dict:
        """
        失敗要因の統計を取得
        
        Args:
            race_date: 日付フィルタ（オプション）
        
        Returns:
            dict: 失敗要因の統計
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if race_date:
                cursor.execute("""
                    SELECT miss_reason_candidates
                    FROM prediction_results
                    WHERE race_date = ? AND miss_reason_candidates IS NOT NULL
                """, (race_date,))
            else:
                cursor.execute("""
                    SELECT miss_reason_candidates
                    FROM prediction_results
                    WHERE miss_reason_candidates IS NOT NULL
                """)
            
            rows = cursor.fetchall()
        
        # 要因タイプ別にカウント
        reason_counts = {}
        for row in rows:
            if row[0]:
                try:
                    reasons = json.loads(row[0])
                    for reason in reasons:
                        reason_type = reason.get('type', 'unknown')
                        reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
                except json.JSONDecodeError:
                    pass
        
        # ソートして返す
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_analyzed': len(rows),
            'reason_counts': dict(sorted_reasons)
        }


def test_analyzer():
    """テスト関数"""
    print("="*60)
    print("🧪 ResultAnalyzer テスト")
    print("="*60)
    
    # 予測ログと結果収集のテストデータを準備
    from src.learning.prediction_logger import PredictionLogger
    from src.learning.result_collector import ResultCollector
    
    logger = PredictionLogger(model_version="2.4-test")
    collector = ResultCollector()
    
    # DBから実際のレースIDを取得
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
    
    # テスト予測データ作成（実際の5着馬を予測1位に = major miss）
    test_data = []
    for i, h in enumerate(horses):
        rank = i + 1
        if i == 0:
            rank = 5  # 実際1着を予測5位に
        elif i == 4:
            rank = 1  # 実際5着を予測1位に
        
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
            'score': 95 - rank * 10,
            'pred_rank': rank,
            'odds_win': h[2],
            'popularity': h[3]
        })
    
    # pred_rankでソート
    test_data.sort(key=lambda x: x['pred_rank'])
    for i, d in enumerate(test_data):
        d['pred_rank'] = i + 1
        d['score'] = 90 - i * 10
    
    test_df = pd.DataFrame(test_data)
    
    # 予測ログ記録
    print()
    logger.log_predictions(test_df, track_condition='重', weather='雨')
    
    # 結果収集
    print()
    collector.compare_and_save(race_ids=[test_race_id])
    
    # 差分分析
    print()
    analyzer = ResultAnalyzer()
    result = analyzer.analyze_results(race_ids=[test_race_id])
    
    print()
    print(f"📊 分析結果:")
    print(f"  分析数: {result['analyzed']}")
    print(f"  スキップ: {result['skipped']}")
    
    # 分析サマリー
    summary = analyzer.get_analysis_summary()
    print()
    print("📈 波乱度分布:")
    print(f"  none (的中): {summary['none']}件 ({summary['none_pct']}%)")
    print(f"  minor (2-3着): {summary['minor']}件 ({summary['minor_pct']}%)")
    print(f"  major (4-6着): {summary['major']}件 ({summary['major_pct']}%)")
    print(f"  extreme (7着以下): {summary['extreme']}件 ({summary['extreme_pct']}%)")
    
    # 失敗要因統計
    miss_stats = analyzer.get_miss_reason_stats()
    print()
    print("📋 失敗要因:")
    for reason_type, count in miss_stats['reason_counts'].items():
        print(f"  {reason_type}: {count}件")
    
    # prediction_results の内容確認
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT race_id, upset_level, miss_reason_candidates,
                   pred_horse_name, actual_1st_name, pred_1st_actual_pos
            FROM prediction_results
            WHERE race_id = ?
        """, (test_race_id,))
        pr_row = cursor.fetchone()
    
    if pr_row:
        print()
        print("📋 prediction_results 詳細:")
        print(f"  波乱度: {pr_row[1]}")
        print(f"  予測1位: {pr_row[3]} → 実際: {pr_row[5]}着")
        print(f"  実際1着: {pr_row[4]}")
        if pr_row[2]:
            reasons = json.loads(pr_row[2])
            print(f"  失敗要因候補:")
            for r in reasons[:3]:
                print(f"    - [{r['severity']}] {r['description']}")
    
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
    test_analyzer()
