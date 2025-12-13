"""
日次ウォークフォワード学習 - LightGBMモデル版
- 2015-2024年のデータで初期モデル構築
- 2025年1月から日次で予測→結果収集→分析→パターン更新
- LightGBMで22特徴量を活用
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from learning.prediction_logger import PredictionLogger
from learning.result_collector import ResultCollector
from learning.result_analyzer import ResultAnalyzer
from learning.pattern_extractor import PatternExtractor
from learning.pattern_validator import PatternValidator
from learning.pattern_applier import PatternApplier

DB_PATH = Path('data/keiba.db')
CSV_PATH = Path('data/csv_imports/results/20150105_20251130all.csv')

# 脚質スコア（勝率ベース）
RUNNING_STYLE_SCORE = {
    '逃げ': 19.5, '先行': 13.9, '中団': 4.9, '後方': 1.4, 'ﾏｸﾘ': 16.3, '不明': 5.0
}

# 前走着順スコア
PREV_FINISH_SCORE = {
    1: 11.2, 2: 20.2, 3: 14.1, 4: 10.4, 5: 7.8,
    6: 6.0, 7: 5.0, 8: 4.0, 9: 3.5, 10: 3.0
}


def zen_to_han(s):
    """全角数字を半角に変換"""
    if pd.isna(s):
        return np.nan
    s = str(s)
    for z, h in zip('０１２３４５６７８９', '0123456789'):
        s = s.replace(z, h)
    try:
        return int(s) if s.isdigit() else np.nan
    except:
        return np.nan


def load_training_data(end_date: str) -> pd.DataFrame:
    """訓練用データを読み込み（指定日まで）"""
    df = pd.read_csv(CSV_PATH, encoding='cp932', low_memory=False)
    
    # 日付変換
    df['date'] = df['日付'].apply(lambda x: f"20{str(x)[:2]}-{str(x)[2:4]}-{str(x)[4:6]}" if pd.notna(x) else None)
    
    # 指定日までフィルタ
    df = df[df['date'] <= end_date]
    
    # 着順変換
    df['finish_position'] = df['着順'].apply(zen_to_han)
    df['prev_finish'] = df['前走着順'].apply(zen_to_han)
    
    # 場所コード
    place_to_code = {'札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
                     '東京': '05', '中山': '06', '中京': '07', '京都': '08',
                     '阪神': '09', '小倉': '10'}
    df['place_code'] = df['場所'].map(place_to_code)
    
    # race_id生成
    df['race_id'] = df['date'] + '_' + df['place_code'] + '_' + df['Ｒ'].astype(str).str.zfill(2)
    
    # カラム名マッピング
    df = df.rename(columns={
        'Ｒ': 'race_no', '馬番': 'horse_no', '馬名': 'horse_name',
        '単勝オッズ': 'odds_win', '人気': 'popularity', '年齢': 'horse_age',
        '性別': 'horse_sex', '馬体重': 'horse_weight', '斤量': 'load_weight',
        '枠番': 'gate_no', '上り3F': 'last_3f_time', '頭数': 'field_size',
        '芝・ダ': 'track_type', '距離': 'distance', '馬場状態': 'track_condition',
        '天気': 'weather', '騎手': 'jockey_name', '調教師': 'trainer_name',
        '脚質': 'running_style', '間隔': 'rest_weeks'
    })
    
    # 数値変換
    for col in ['odds_win', 'popularity', 'horse_age', 'horse_weight', 
                'load_weight', 'gate_no', 'last_3f_time', 'field_size', 'distance', 'rest_weeks']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # track_type変換
    df['track_type'] = df['track_type'].replace({'ダ': 'ダート'})
    
    # 有効データのみ
    df = df[df['finish_position'].notna() & (df['finish_position'] > 0)]
    
    # スコア追加
    df['running_style_score'] = df['running_style'].map(RUNNING_STYLE_SCORE).fillna(5.0)
    df['prev_finish_score'] = df['prev_finish'].map(PREV_FINISH_SCORE).fillna(2.0)
    df['rest_weeks'] = df['rest_weeks'].clip(0, 52)
    df['rest_weeks_optimal'] = ((df['rest_weeks'] >= 2) & (df['rest_weeks'] <= 4)).astype(int)
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を作成"""
    features = pd.DataFrame(index=df.index)
    
    # 基本特徴量
    features['odds_log'] = np.log1p(df['odds_win'].fillna(100))
    features['popularity'] = df['popularity'].fillna(10)
    features['horse_age'] = df['horse_age'].fillna(4)
    features['load_weight'] = df['load_weight'].fillna(55)
    features['gate_no'] = df['gate_no'].fillna(4)
    features['field_size'] = df['field_size'].fillna(14)
    features['distance'] = df['distance'].fillna(1600)
    
    # 相対特徴量
    features['popularity_ratio'] = features['popularity'] / features['field_size']
    features['odds_rank'] = df.groupby('race_id')['odds_win'].rank(method='min').fillna(10)
    
    # カテゴリカル特徴量
    features['is_turf'] = (df['track_type'] == '芝').astype(int)
    features['is_good_track'] = (df['track_condition'] == '良').astype(int)
    
    # 性別
    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    features['horse_sex'] = df['horse_sex'].map(sex_map).fillna(0)
    
    # 場所コード
    features['place_code'] = df['place_code'].fillna('05').astype(str).str.zfill(2)
    features['place_code'] = pd.Categorical(features['place_code']).codes
    
    # 新特徴量
    features['running_style_score'] = df['running_style_score'].fillna(5.0)
    features['prev_finish_score'] = df['prev_finish_score'].fillna(2.0)
    features['rest_weeks'] = df['rest_weeks'].fillna(4)
    features['rest_weeks_optimal'] = df['rest_weeks_optimal'].fillna(0)
    
    # 上がり3F
    features['last_3f_time'] = df['last_3f_time'].iloc[:, 0].fillna(35.0) if isinstance(df['last_3f_time'], pd.DataFrame) else df['last_3f_time'].fillna(35.0)
    
    # 馬体重
    features['horse_weight'] = df['horse_weight'].iloc[:, 0].fillna(480) if isinstance(df['horse_weight'], pd.DataFrame) else df['horse_weight'].fillna(480)
    
    return features


class WalkForwardLearning:
    """日次ウォークフォワード学習（LightGBM版）"""
    
    def __init__(self):
        self.results_log = []
        self.model = None
        self.feature_cols = None
        
    def clear_learning_tables(self):
        """学習テーブルをクリア"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for t in ['prediction_logs', 'prediction_results', 'pattern_candidates', 'validated_patterns']:
            cursor.execute(f'DELETE FROM {t}')
        conn.commit()
        conn.close()
        print("✅ 学習テーブルクリア完了")
    
    def train_model(self, train_end_date: str):
        """LightGBMモデルを訓練"""
        print(f"\n📚 モデル訓練（〜{train_end_date}）")
        
        # データ読み込み
        df = load_training_data(train_end_date)
        print(f"  訓練データ: {len(df):,}件")
        
        # 特徴量作成
        features = create_features(df)
        self.feature_cols = features.columns.tolist()
        
        # ターゲット（1着=1, その他=0）
        y = (df['finish_position'] == 1).astype(int)
        X = features
        
        # 訓練/検証分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # LightGBMモデル
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # 特徴量重要度
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print(f"  → モデル訓練完了")
        print(f"  → Top5特徴量: {', '.join(importance.head(5)['feature'].tolist())}")
        
        return self.model
    
    def predict_for_date(self, date: str) -> pd.DataFrame:
        """指定日のレースを予測"""
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"""
            SELECT race_id, date, race_no, horse_no, horse_name, finish_position,
                   odds_win, popularity, place_code, distance, track_type,
                   track_condition, weather, field_size, horse_age, horse_sex,
                   load_weight, gate_no, running_style, rest_weeks, last_3f_time,
                   horse_weight, last_3f_time,
                   horse_weight
            FROM race_results WHERE date = '{date}'
            ORDER BY race_id, horse_no
        """, conn)
        conn.close()
        
        if df.empty:
            return df
        
        # スコア追加
        df['running_style_score'] = df['running_style'].map(RUNNING_STYLE_SCORE).fillna(5.0)
        df['prev_finish_score'] = 5.0  # 簡易版
        df['rest_weeks'] = df['rest_weeks'].clip(0, 52) if 'rest_weeks' in df.columns else 4
        df['rest_weeks_optimal'] = ((df['rest_weeks'] >= 2) & (df['rest_weeks'] <= 4)).astype(int)
        
        # 特徴量作成
        features = create_features(df)
        
        # 予測
        df['pred_prob'] = self.model.predict(features[self.feature_cols])
        df['score'] = df['pred_prob'] * 100
        
        # パターン適用
        applier = PatternApplier()
        if applier.patterns:
            df, _ = applier.apply_patterns(df)
        
        df['pred_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first').astype(int)
        
        return df
    
    def run_daily_cycle(self, date: str, logger) -> dict:
        """1日分の学習サイクル"""
        result = {
            'date': date,
            'races': 0,
            'predictions': 0,
            'hit_1st': 0,
            'hit_top3': 0
        }
        
        # 予測
        df = self.predict_for_date(date)
        if df.empty:
            return result
        
        result['predictions'] = len(df)
        
        # ログ記録
        for race_id in df['race_id'].unique():
            race_df = df[df['race_id'] == race_id].copy()
            features_df = pd.DataFrame({'odds_log': np.log1p(race_df['odds_win'].fillna(100))})
            track_cond = race_df['track_condition'].iloc[0] if pd.notna(race_df['track_condition'].iloc[0]) else '良'
            weather = race_df['weather'].iloc[0] if pd.notna(race_df['weather'].iloc[0]) else '晴'
            logger.log_predictions(race_df, features_df, [], track_cond, weather)
        
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
        
        return result
    
    def run_walk_forward(self, start_date: str = '2025-01-01', end_date: str = '2025-11-30',
                         retrain_frequency: int = 30):
        """ウォークフォワード学習を実行"""
        print("="*60)
        print("🚀 日次ウォークフォワード学習（LightGBM）")
        print(f"   期間: {start_date} 〜 {end_date}")
        print(f"   モデル再訓練頻度: {retrain_frequency}日ごと")
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
        
        # 初期モデル訓練
        self.train_model('2024-12-31')
        
        logger = PredictionLogger(model_version='wf-lgbm')
        monthly_stats = {}
        
        for i, date in enumerate(dates):
            # モデル再訓練（指定日数ごと）
            if i > 0 and i % retrain_frequency == 0:
                prev_date = dates[i-1]
                self.train_model(prev_date)
            
            result = self.run_daily_cycle(date, logger)
            self.results_log.append(result)
            
            # 月別集計
            month = date[:7]
            if month not in monthly_stats:
                monthly_stats[month] = {'races': 0, 'hit_1st': 0, 'hit_top3': 0}
            monthly_stats[month]['races'] += result['races']
            monthly_stats[month]['hit_1st'] += result['hit_1st']
            monthly_stats[month]['hit_top3'] += result['hit_top3']
            
            # 進捗表示
            if (i + 1) % 20 == 0:
                print(f"  処理中: {date} ({i+1}/{len(dates)})")
        
        # 結果表示
        print("\n" + "="*60)
        print("📊 月別精度推移（LightGBM）")
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
        
        total_races = sum(s['races'] for s in monthly_stats.values())
        total_hit_1st = sum(s['hit_1st'] for s in monthly_stats.values())
        total_hit_top3 = sum(s['hit_top3'] for s in monthly_stats.values())
        
        print("-"*70)
        print(f"{'合計':^10} | {total_races:^8} | {total_hit_1st:^8} | {total_hit_1st/total_races*100:^7.1f}% | {total_hit_top3:^8} | {total_hit_top3/total_races*100:^7.1f}%")
        
        return monthly_stats


def main():
    wf = WalkForwardLearning()
    
    # 1. 学習テーブルクリア
    wf.clear_learning_tables()
    
    # 2. ウォークフォワード学習実行
    wf.run_walk_forward(
        start_date='2025-01-05',
        end_date='2025-11-30',
        retrain_frequency=30  # 月1回モデル再訓練
    )
    
    print("\n✅ ウォークフォワード学習完了")


if __name__ == "__main__":
    main()
