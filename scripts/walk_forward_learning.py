"""
日次ウォークフォワード学習
- 2015-2024年のデータで初期パターン構築
- 2025年1月から日次で予測→結果収集→分析→パターン更新
- 精度推移を記録
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from learning.prediction_logger import PredictionLogger
from learning.result_collector import ResultCollector
from learning.result_analyzer import ResultAnalyzer
from learning.pattern_extractor import PatternExtractor
from learning.pattern_validator import PatternValidator
from learning.pattern_applier import PatternApplier

DB_PATH = Path('data/keiba.db')

class WalkForwardLearning:
    """日次ウォークフォワード学習"""
    
    def __init__(self):
        self.results_log = []  # 日次結果を記録
        
    def clear_learning_tables(self):
        """学習テーブルをクリア"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for t in ['prediction_logs', 'prediction_results', 'pattern_candidates', 'validated_patterns']:
            cursor.execute(f'DELETE FROM {t}')
        conn.commit()
        conn.close()
        print("✅ 学習テーブルクリア完了")
    
    def build_initial_patterns(self, end_date: str = '2024-12-31'):
        """初期パターン構築（2015-2024年データ）"""
        print("="*60)
        print("📚 初期パターン構築（2015-2024年）")
        print("="*60)
        
        conn = sqlite3.connect(DB_PATH)
        
        # 2024年までのデータで予測シミュレーション
        dates = pd.read_sql(f"""
            SELECT DISTINCT date FROM race_results 
            WHERE date >= '2024-01-01' AND date <= '{end_date}'
            ORDER BY date
        """, conn)['date'].tolist()
        conn.close()
        
        print(f"  訓練データ期間: 2024-01-01 〜 {end_date}")
        print(f"  対象日数: {len(dates)}日")
        
        # 予測シミュレーション
        logger = PredictionLogger(model_version='wf-init')
        total_logged = 0
        
        for date in dates:
            logged = self._simulate_predictions_for_date(date, logger)
            total_logged += logged
        
        print(f"  → 予測ログ: {total_logged:,}件")
        
        # 結果収集
        collector = ResultCollector()
        for date in dates:
            pending = collector.get_pending_races(date)
            if pending:
                collector.compare_and_save(race_ids=pending, race_date=date)
        
        # 分析
        analyzer = ResultAnalyzer()
        for date in dates:
            unanalyzed = analyzer.get_unanalyzed_results(date)
            if unanalyzed:
                analyzer.analyze_results(race_ids=unanalyzed, race_date=date)
        
        # パターン抽出・検証
        extractor = PatternExtractor(min_sample_size=100, min_effect_size=3.0)
        extractor.extract_all()
        
        validator = PatternValidator(min_sample_size=500, min_effect_size=1.0)
        result = validator.validate_all(start_date='2015-01-01')
        
        print(f"  → 検証合格パターン: {result.get('validated', 0)}件")
        
        return result.get('validated', 0)
    
    def _simulate_predictions_for_date(self, date: str, logger) -> int:
        """指定日の予測をシミュレート"""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"""
            SELECT race_id, date, race_no, horse_no, horse_name, finish_position,
                   odds_win, popularity, place_code, distance, track_type,
                   track_condition, weather, field_size, horse_age, horse_sex,
                   load_weight, gate_no, running_style, rest_weeks
            FROM race_results WHERE date = '{date}'
            ORDER BY race_id, horse_no
        """, conn)
        conn.close()
        
        if df.empty:
            return 0
        
        np.random.seed(hash(date) % 2**32)
        
        # 基本スコア（人気ベース）
        df['base_score'] = 100 - (df['popularity'].fillna(10) * 5)
        df['base_score'] += np.random.normal(0, 5, len(df))
        
        # パターン適用
        df['score'] = df['base_score'].clip(0, 100)
        
        applier = PatternApplier()
        if applier.patterns:
            df, _ = applier.apply_patterns(df)
        
        df['pred_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first').astype(int)
        
        # ログ記録
        logged = 0
        for race_id in df['race_id'].unique():
            race_df = df[df['race_id'] == race_id].copy()
            features_df = pd.DataFrame({'odds_log': np.log1p(race_df['odds_win'].fillna(100))})
            track_cond = race_df['track_condition'].iloc[0] if pd.notna(race_df['track_condition'].iloc[0]) else '良'
            weather = race_df['weather'].iloc[0] if pd.notna(race_df['weather'].iloc[0]) else '晴'
            result = logger.log_predictions(race_df, features_df, [], track_cond, weather)
            logged += result.get('logged', 0)
        
        return logged
    
    def run_daily_cycle(self, date: str, logger, update_patterns: bool = True) -> dict:
        """1日分の学習サイクル"""
        result = {
            'date': date,
            'races': 0,
            'predictions': 0,
            'hit_1st': 0,
            'hit_top3': 0,
            'patterns_before': 0,
            'patterns_after': 0
        }
        
        # 現在のパターン数
        conn = sqlite3.connect(DB_PATH)
        result['patterns_before'] = pd.read_sql(
            "SELECT COUNT(*) as c FROM validated_patterns WHERE is_active=1", conn
        ).iloc[0, 0]
        conn.close()
        
        # 予測シミュレーション
        logged = self._simulate_predictions_for_date(date, logger)
        result['predictions'] = logged
        
        if logged == 0:
            return result
        
        # 結果収集
        collector = ResultCollector()
        pending = collector.get_pending_races(date)
        if pending:
            collector.compare_and_save(race_ids=pending, race_date=date)
            result['races'] = len(pending)
        
        # 的中率計算
        conn = sqlite3.connect(DB_PATH)
        accuracy = pd.read_sql(f"""
            SELECT 
                SUM(is_hit_1st) as hit_1st,
                SUM(is_hit_top3) as hit_top3,
                COUNT(*) as total
            FROM prediction_results
            WHERE race_date = '{date}'
        """, conn)
        conn.close()
        
        if accuracy['total'].iloc[0] > 0:
            result['hit_1st'] = int(accuracy['hit_1st'].iloc[0] or 0)
            result['hit_top3'] = int(accuracy['hit_top3'].iloc[0] or 0)
        
        # 分析
        analyzer = ResultAnalyzer()
        unanalyzed = analyzer.get_unanalyzed_results(date)
        if unanalyzed:
            analyzer.analyze_results(race_ids=unanalyzed, race_date=date)
        
        # パターン更新（週単位など）
        if update_patterns:
            extractor = PatternExtractor(min_sample_size=50, min_effect_size=3.0)
            extractor.extract_all()
            
            validator = PatternValidator(min_sample_size=300, min_effect_size=1.0)
            validator.validate_all(start_date='2015-01-01')
        
        # 更新後のパターン数
        conn = sqlite3.connect(DB_PATH)
        result['patterns_after'] = pd.read_sql(
            "SELECT COUNT(*) as c FROM validated_patterns WHERE is_active=1", conn
        ).iloc[0, 0]
        conn.close()
        
        return result
    
    def run_walk_forward(self, start_date: str = '2025-01-01', end_date: str = '2025-11-30',
                         update_frequency: int = 7):
        """ウォークフォワード学習を実行"""
        print("="*60)
        print("🚀 日次ウォークフォワード学習開始")
        print(f"   期間: {start_date} 〜 {end_date}")
        print(f"   パターン更新頻度: {update_frequency}日ごと")
        print("="*60)
        
        # 対象日取得
        conn = sqlite3.connect(DB_PATH)
        dates = pd.read_sql(f"""
            SELECT DISTINCT date FROM race_results 
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date
        """, conn)['date'].tolist()
        conn.close()
        
        print(f"  対象日数: {len(dates)}日")
        
        logger = PredictionLogger(model_version='wf-daily')
        
        # 月別集計用
        monthly_stats = {}
        
        for i, date in enumerate(dates):
            # パターン更新は指定日数ごと
            update_patterns = (i % update_frequency == 0)
            
            result = self.run_daily_cycle(date, logger, update_patterns)
            self.results_log.append(result)
            
            # 月別集計
            month = date[:7]
            if month not in monthly_stats:
                monthly_stats[month] = {'races': 0, 'hit_1st': 0, 'hit_top3': 0}
            monthly_stats[month]['races'] += result['races']
            monthly_stats[month]['hit_1st'] += result['hit_1st']
            monthly_stats[month]['hit_top3'] += result['hit_top3']
            
            # 進捗表示（10日ごと）
            if (i + 1) % 10 == 0:
                print(f"  処理中: {date} ({i+1}/{len(dates)})")
        
        print("\n" + "="*60)
        print("📊 月別精度推移")
        print("="*60)
        print(f"{'月':^10} | {'レース':^8} | {'1着的中':^8} | {'的中率':^8} | {'Top3的中':^8} | {'Top3率':^8}")
        print("-"*70)
        
        for month, stats in sorted(monthly_stats.items()):
            races = stats['races']
            hit_1st = stats['hit_1st']
            hit_top3 = stats['hit_top3']
            rate_1st = hit_1st / races * 100 if races > 0 else 0
            rate_top3 = hit_top3 / races * 100 if races > 0 else 0
            print(f"{month:^10} | {races:^8} | {hit_1st:^8} | {rate_1st:^7.1f}% | {hit_top3:^8} | {rate_top3:^7.1f}%")
        
        # 全体集計
        total_races = sum(s['races'] for s in monthly_stats.values())
        total_hit_1st = sum(s['hit_1st'] for s in monthly_stats.values())
        total_hit_top3 = sum(s['hit_top3'] for s in monthly_stats.values())
        
        print("-"*70)
        print(f"{'合計':^10} | {total_races:^8} | {total_hit_1st:^8} | {total_hit_1st/total_races*100:^7.1f}% | {total_hit_top3:^8} | {total_hit_top3/total_races*100:^7.1f}%")
        
        return monthly_stats
    
    def save_results(self, filepath: str = 'temp/walk_forward_results.json'):
        """結果をファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results_log, f, ensure_ascii=False, indent=2)
        print(f"\n💾 結果保存: {filepath}")


def main():
    wf = WalkForwardLearning()
    
    # 1. 学習テーブルクリア
    wf.clear_learning_tables()
    
    # 2. 初期パターン構築（2024年データ）
    wf.build_initial_patterns(end_date='2024-12-31')
    
    # 3. 2025年1月からウォークフォワード学習
    wf.run_walk_forward(
        start_date='2025-01-01',
        end_date='2025-11-30',
        update_frequency=7  # 週1回パターン更新
    )
    
    # 4. 結果保存
    wf.save_results()
    
    print("\n✅ ウォークフォワード学習完了")


if __name__ == "__main__":
    main()
