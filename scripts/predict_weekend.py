"""
改善版オッズ統合予測スクリプト v2.4 - Keiba Intelligence
3カテゴリー推奨レース版（鉄板・中穴・大穴）
- 人気の盲点（ギャップ指標）による選定
- TOP4馬券種対応（単勝・馬連・3連複・3連単）
"""
import pandas as pd
import numpy as np
import sqlite3
import lightgbm as lgb
import json
from pathlib import Path
from datetime import datetime
import argparse
import warnings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from learning.prediction_logger import PredictionLogger
warnings.filterwarnings('ignore')

# ============================================================
# 設定
# ============================================================
DB_PATH = Path('data/keiba.db')
CSV_DIR = Path('data/csv_imports')
ODDS_DIR = Path('data/csv_imports/odds')
ENTRY_DIR = Path('data/csv_imports/race card')
RESULTS_DIR = Path('data/csv_imports/results')
PREDICTION_DIR = Path('data/predictions')

# カテゴリー定義（配当期待値ベース）
CATEGORY_CONFIG = {
    'TEPPAN': {
        'name': '鉄板',
        'emoji': '🔵',
        'description': '高確率で的中を狙う',
        'bet_style': '堅実型',
        'target': '単勝100〜350円',
        # 選定条件: AIランク1位 × 1〜2番人気 × オッズ1.0〜3.5倍
        'rank_max': 1,
        'popularity_max': 2,
        'odds_min': 1.0,
        'odds_max': 3.5,
    },
    'CHUUANA': {
        'name': '中穴',
        'emoji': '🟡',
        'description': '馬連2,000円以上を狙う',
        'bet_style': 'バランス型',
        'target': '馬連2,000〜10,000円',
        # 選定条件: AIランク1〜3位 × 3〜7番人気 × オッズ5〜20倍 × ギャップ+2以上
        'rank_max': 3,
        'popularity_min': 3,
        'popularity_max': 7,
        'odds_min': 5.0,
        'odds_max': 20.0,
        'gap_min': 2,
    },
    'OOANA': {
        'name': '大穴',
        'emoji': '🔴',
        'description': '万馬券を狙う',
        'bet_style': '高配当型',
        'target': '馬連10,000円以上',
        # 選定条件: AIランク1〜5位 × 8番人気以下 × オッズ20倍以上 × ギャップ+3以上
        'rank_max': 5,
        'popularity_min': 8,
        'odds_min': 20.0,
        'gap_min': 3,
    }
}

# 買い目点数設定
BET_CONFIG = {
    'TEPPAN': {
        'tansho': 1,      # ◎のみ
        'umaren': 1,      # ◎-○
        'sanrenpuku': 6,  # ◎-○▲-○▲△ 
        'sanrentan': 6,   # ◎→○▲→○▲△
    },
    'CHUUANA': {
        'tansho': 1,
        'umaren': 2,      # ◎-○▲
        'sanrenpuku': 12, # ◎-○▲△-○▲△☆
        'sanrentan': 12,  # ◎→○▲→○▲△△
    },
    'OOANA': {
        'tansho': 1,
        'umaren': 3,      # ◎-○▲△
        'sanrenpuku': 19, # 2-4-7形式
        'sanrentan': 24,  # ◎○→◎○▲→○▲△△△
    }
}

PLACE_CSV_TO_CODE = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
    '東京': '05', '中山': '06', '中京': '07', '京都': '08',
    '阪神': '09', '小倉': '10'
}

PLACE_CODE_TO_NAME = {v: k for k, v in PLACE_CSV_TO_CODE.items() if len(k) == 2}
TRACK_TYPE_MAP = {'芝': '芝', 'ダ': 'ダート', 'ダート': 'ダート'}


def detect_race_class(race_name):
    race_name = str(race_name)
    if 'G1' in race_name or 'ＧＩ' in race_name:
        return 'G1'
    elif 'G2' in race_name or 'ＧＩＩ' in race_name:
        return 'G2'
    elif 'G3' in race_name or 'ＧＩＩ' in race_name:
        return 'G3'
    elif '新馬' in race_name:
        return '新馬'
    elif '未勝利' in race_name:
        return '未勝利'
    elif '1勝' in race_name or '１勝' in race_name:
        return '1勝'
    elif '2勝' in race_name or '２勝' in race_name:
        return '2勝'
    elif '3勝' in race_name or '３勝' in race_name:
        return '3勝'
    elif 'オープン' in race_name or 'OP' in race_name:
        return 'OP'
    return '条件'


def is_graded_race(race_class):
    return race_class in ['G1', 'G2', 'G3']


def is_maiden_race(race_class):
    return race_class in ['新馬', '未勝利']


def parse_odds_key(key):
    key = str(key).strip()
    place_first = key[0]
    place_map = {'1': '01', '2': '02', '3': '03', '4': '04', '5': '05',
                 '6': '06', '7': '07', '8': '08', '9': '09', '0': '10'}
    place_code = place_map.get(place_first, '05')

    if len(key) == 10:
        race_no = int(key[5:7])
        horse_no = int(key[7:10])
    elif len(key) == 9:
        race_no = int(key[5:7])
        horse_no = int(key[7:9])
    else:
        race_no, horse_no = 0, 0
    return place_code, race_no, horse_no


def load_odds_csv(date_str):
    short_date = date_str[2:]
    odds_path = ODDS_DIR / f"OD{short_date}.CSV"
    if not odds_path.exists():
        print(f"⚠️  オッズファイルなし: {odds_path}")
        return None

    print(f"📊 オッズCSV読み込み: {odds_path}")
    df = pd.read_csv(odds_path, header=None, encoding='cp932')

    odds_df = pd.DataFrame({
        'race_key': df[0].astype(str).str.strip(),
        'odds_win': pd.to_numeric(df[7], errors='coerce'),
        'odds_place_low': pd.to_numeric(df[8], errors='coerce'),
        'odds_place_high': pd.to_numeric(df[9], errors='coerce')
    })

    parsed = odds_df['race_key'].apply(
        lambda x: pd.Series(parse_odds_key(x), index=['place_code', 'race_no', 'horse_no'])
    )
    odds_df = pd.concat([odds_df, parsed], axis=1)
    print(f"  → {len(odds_df)}件")
    return odds_df


def load_entry_csv(date_str):
    short_date = date_str[2:]
    entry_path = ENTRY_DIR / f"DE{short_date}.CSV"
    if not entry_path.exists():
        print(f"❌ 出馬表ファイルなし")
        return None

    print(f"📄 出馬表CSV読み込み: {entry_path}")
    df = pd.read_csv(entry_path, encoding='cp932')
    print(f"  → {len(df)}頭")
    return df


def merge_entry_and_odds(entry_df, odds_df):
    entry_df['place_code'] = entry_df['場所'].map(PLACE_CSV_TO_CODE)
    entry_df['race_no'] = entry_df['R'].astype(int)
    entry_df['horse_no'] = entry_df['馬番'].astype(int)

    entry_df['merge_key'] = (
        entry_df['place_code'] + '_' +
        entry_df['race_no'].astype(str).str.zfill(2) + '_' +
        entry_df['horse_no'].astype(str).str.zfill(2)
    )

    odds_df['merge_key'] = (
        odds_df['place_code'] + '_' +
        odds_df['race_no'].astype(str).str.zfill(2) + '_' +
        odds_df['horse_no'].astype(str).str.zfill(2)
    )

    merged = entry_df.merge(
        odds_df[['merge_key', 'odds_win', 'odds_place_low', 'odds_place_high']],
        on='merge_key', how='left'
    )

    odds_count = merged['odds_win'].notna().sum()
    print(f"  → オッズ統合: {odds_count}/{len(merged)}件")
    return merged


def convert_to_predict_format(merged_df, date_str):
    date_formatted = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"

    records = []
    for _, row in merged_df.iterrows():
        place_code = row['place_code']
        race_no = int(row['race_no'])
        race_name = str(row['レース名'])
        race_class = detect_race_class(race_name)
        race_id = f"{date_formatted}_{place_code}_{race_no:02d}"

        record = {
            'race_id': race_id, 'date': date_formatted,
            'place_code': place_code,
            'place_name': PLACE_CODE_TO_NAME.get(place_code, '不明'),
            'race_no': race_no, 'race_name': race_name,
            'race_class': race_class,
            'is_graded': is_graded_race(race_class),
            'is_maiden': is_maiden_race(race_class),
            'track_type': TRACK_TYPE_MAP.get(row['芝・ダ'], row['芝・ダ']),
            'distance': int(row['距離']),
            'horse_no': int(row['馬番']),
            'horse_name': str(row['馬名']).strip(),
            'horse_sex': row['性別'],
            'horse_age': int(row['年齢']),
            'jockey_name': str(row['騎手']).strip(),
            'load_weight': float(row['斤量']),
            'trainer_name': str(row['調教師']).strip(),
            'odds_win': float(row['odds_win']) if pd.notna(row['odds_win']) else None,
            'odds_place_low': float(row['odds_place_low']) if pd.notna(row.get('odds_place_low')) else None,
            'gate_no': int(row['枠']) if pd.notna(row.get('枠')) and row.get('枠') != 0 else None
        }
        records.append(record)

    pred_df = pd.DataFrame(records)
    field_sizes = pred_df.groupby('race_id').size().reset_index(name='field_size')
    pred_df = pred_df.merge(field_sizes, on='race_id')
    return pred_df




def zen_to_han(s):
    """全角数字を半角に変換"""
    if pd.isna(s):
        return np.nan
    s = str(s)
    zen = '０１２３４５６７８９'
    han = '0123456789'
    for z, h in zip(zen, han):
        s = s.replace(z, h)
    try:
        return int(s) if s.isdigit() else np.nan
    except:
        return np.nan


def load_historical_data_from_csv():
    """新CSVから過去データを読み込み（脚質・前走情報付き）"""
    print("📚 過去データ読み込み中（新CSV）...")
    
    csv_path = CSV_DIR / 'results' / '20150105_20251130all.csv'
    
    if not csv_path.exists():
        print(f"⚠️  CSVファイルが見つかりません: {csv_path}")
        return load_historical_data()  # フォールバック
    
    # 必要なカラムのみ読み込み
    use_cols = [
        '日付', '場所', 'Ｒ', '馬番', '馬名', '着順', '単勝オッズ', '人気',
        '年齢', '性別', '騎手', '調教師', '馬体重', '斤量', '枠番', '上り3F',
        '頭数', '芝・ダ', '距離', '馬場状態', '天気', 'クラス名',
        '脚質', '前走着順', '前走脚質', '間隔'
    ]
    
    df = pd.read_csv(csv_path, encoding='cp932', usecols=use_cols, low_memory=False)
    
    # カラム名変換
    df = df.rename(columns={
        '日付': 'date_raw',
        '場所': 'place_name',
        'Ｒ': 'race_no',
        '馬番': 'horse_no',
        '馬名': 'horse_name',
        '着順': 'finish_position_raw',
        '単勝オッズ': 'odds_win',
        '人気': 'popularity',
        '年齢': 'horse_age',
        '性別': 'horse_sex',
        '騎手': 'jockey_name',
        '調教師': 'trainer_name',
        '馬体重': 'horse_weight',
        '斤量': 'load_weight',
        '枠番': 'gate_no',
        '上り3F': 'last_3f_time',
        '頭数': 'field_size',
        '芝・ダ': 'track_type',
        '距離': 'distance',
        '馬場状態': 'track_condition',
        '天気': 'weather',
        'クラス名': 'race_class',
        '脚質': 'running_style',
        '前走着順': 'prev_finish_raw',
        '前走脚質': 'prev_running_style',
        '間隔': 'rest_weeks'
    })
    
    # データ変換
    # 日付変換 (YYMMDD → YYYY-MM-DD)
    df['date'] = df['date_raw'].apply(lambda x: f"20{str(x)[:2]}-{str(x)[2:4]}-{str(x)[4:6]}" if pd.notna(x) else None)
    
    # 着順変換（全角→半角）
    df['finish_position'] = df['finish_position_raw'].apply(zen_to_han)
    
    # 前走着順変換（全角→半角）
    df['prev_finish'] = df['prev_finish_raw'].apply(zen_to_han)
    
    # 天気のtrim
    df['weather'] = df['weather'].str.strip() if df['weather'].dtype == 'object' else df['weather']
    
    # トラックタイプ変換
    df['track_type'] = df['track_type'].replace({'ダ': 'ダート'})
    
    # 場所コード追加
    place_to_code = {
        '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
        '東京': '05', '中山': '06', '中京': '07', '京都': '08',
        '阪神': '09', '小倉': '10'
    }
    df['place_code'] = df['place_name'].map(place_to_code)
    
    # race_id生成
    df['race_id'] = df['date'] + '_' + df['place_code'] + '_' + df['race_no'].astype(str).str.zfill(2)
    
    # 脚質スコア（勝率ベース）
    running_style_score = {
        '逃げ': 19.5,
        '先行': 13.9,
        '中団': 4.9,
        '後方': 1.4,
        'ﾏｸﾘ': 16.3,
        '不明': 5.0
    }
    df['running_style_score'] = df['running_style'].map(running_style_score).fillna(5.0)
    
    # 前走着順スコア
    prev_finish_score = {
        1: 11.2, 2: 20.2, 3: 14.1, 4: 10.4, 5: 7.8,
        6: 6.0, 7: 5.0, 8: 4.0, 9: 3.5, 10: 3.0
    }
    df['prev_finish_score'] = df['prev_finish'].map(prev_finish_score).fillna(5.0)
    
    # 数値型変換
    numeric_cols = ['odds_win', 'popularity', 'horse_age', 'horse_weight', 
                    'load_weight', 'gate_no', 'last_3f_time', 'field_size', 'distance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 有効データのみ抽出
    df = df[
        (df['finish_position'].notna()) & 
        (df['finish_position'] > 0) &
        (df['odds_win'].notna()) & 
        (df['odds_win'] > 0)
    ]
    
    # レースクラス判定
    df['is_graded'] = df['race_class'].str.contains('Ｇ', na=False)
    df['is_maiden'] = df['race_class'].str.contains('新馬|未勝利', na=False)
    
    print(f"  → {len(df):,}件（脚質・前走情報付き）")
    
    return df

def load_historical_data():
    print("📚 過去データ読み込み中...")
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT rr.race_id, rr.date, rr.place_code, rr.race_no,
            rr.horse_no, rr.horse_name, rr.horse_age, rr.horse_sex,
            rr.jockey_name, rr.trainer_name, rr.load_weight, rr.gate_no,
            rr.odds_win, rr.popularity, rr.finish_position,
            rr.field_size, rr.last_3f_time, rr.time,
            rd.distance, rd.track_type, rd.track_condition, rd.weather, rd.race_class
        FROM race_results rr
        LEFT JOIN race_details rd ON rr.race_id = rd.race_id
        WHERE rr.finish_position IS NOT NULL AND rr.finish_position > 0
          AND rr.odds_win IS NOT NULL AND rr.odds_win > 0
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if 'race_class' not in df.columns or df['race_class'].isna().all():
        df['race_class'] = '条件'

    print(f"  → {len(df):,}件")
    return df


def create_features(df, is_prediction=False):
    features = pd.DataFrame()

    features['odds_log'] = np.log1p(df['odds_win'].fillna(100))
    features['odds_place_log'] = np.log1p(df['odds_place_low'].fillna(50)) if 'odds_place_low' in df.columns else features['odds_log'] * 0.3
    features['load_weight'] = df['load_weight'].fillna(55)
    features['weight_ratio'] = df['load_weight'] / 55.0
    features['horse_age'] = df['horse_age'].fillna(3)

    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    features['horse_sex_num'] = df['horse_sex'].map(sex_map).fillna(0)

    track_map = {'芝': 0, 'ダート': 1}
    features['track_type_num'] = df['track_type'].map(track_map).fillna(0)

    condition_map = {'良': 0, '稀重': 1, '重': 2, '不良': 3}
    features['track_condition_num'] = df['track_condition'].map(condition_map).fillna(0)

    weather_map = {'晴': 0, '曇': 1, '小雨': 2, '雨': 3, '小雪': 4, '雪': 5}
    features['weather_num'] = df['weather'].map(weather_map).fillna(0)

    features['distance'] = df['distance'].fillna(1600)
    features['distance_cat'] = pd.cut(df['distance'], bins=[0, 1200, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)

    features['gate_no'] = df['gate_no'].fillna(4)
    features['horse_no'] = df['horse_no'].fillna(8)
    features['field_size'] = df['field_size'].fillna(16)
    features['horse_no_ratio'] = df['horse_no'] / df['field_size'].replace(0, 1)
    features['is_outside'] = (features['horse_no_ratio'] >= 0.8).astype(int)

    features['is_graded'] = df['is_graded'].astype(int) if 'is_graded' in df.columns else 0
    features['is_maiden'] = df['is_maiden'].astype(int) if 'is_maiden' in df.columns else 0


    # === 新特徴量（脚質・前走情報）===
    if 'running_style_score' in df.columns:
        features['running_style_score'] = df['running_style_score'].fillna(5.0)
    else:
        features['running_style_score'] = 5.0

    if 'prev_finish_score' in df.columns:
        features['prev_finish_score'] = df['prev_finish_score'].fillna(5.0)
    else:
        features['prev_finish_score'] = 5.0

    if 'rest_weeks' in df.columns:
        features['rest_weeks'] = df['rest_weeks'].fillna(4).clip(0, 52)
        features['rest_weeks_optimal'] = ((df['rest_weeks'] >= 2) & (df['rest_weeks'] <= 4)).astype(int)
    else:
        features['rest_weeks'] = 4
        features['rest_weeks_optimal'] = 1

    return features


def train_model(hist_df):
    print("🤖 モデル訓練中...")

    hist_df['is_graded'] = hist_df['race_class'].apply(lambda x: is_graded_race(str(x)) if pd.notna(x) else False)
    hist_df['is_maiden'] = hist_df['race_class'].apply(lambda x: is_maiden_race(str(x)) if pd.notna(x) else False)

    X = create_features(hist_df)
    y = (hist_df['finish_position'] <= 3).astype(int)

    feature_cols = X.columns.tolist()

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.02, num_leaves=25, max_depth=6,
        min_child_samples=50, lambda_l1=0.1, lambda_l2=0.2,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        random_state=42, verbose=-1
    )
    model.fit(X, y)
    print(f"  → 特徴量数: {len(feature_cols)}")
    return model, feature_cols


def predict_races(model, feature_cols, pred_df):
    print("🎯 予測実行中...")

    df = pred_df.copy()
    df['track_condition'] = '良'
    df['weather'] = '晴'

    X_pred = create_features(df, is_prediction=True)

    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0

    X_pred = X_pred[feature_cols]
    probs = model.predict_proba(X_pred)[:, 1]
    df['pred_prob'] = probs
    df['score'] = df['pred_prob'] * 100

    # AIスコアによるランク
    df['pred_rank'] = df.groupby('race_id')['pred_prob'].rank(ascending=False, method='first').astype(int)

    # 人気順（オッズベース）
    if df['odds_win'].notna().any():
        df['popularity'] = df.groupby('race_id')['odds_win'].rank(method='first').astype(int)
    else:
        df['popularity'] = df['pred_rank']  # オッズがない場合はAIランクで代用

    # ギャップ指標 = 人気順 - AIランク（プラス = 過小評価 = 狙い目）
    df['gap'] = df['popularity'] - df['pred_rank']

    return df


def select_recommended_races(result_df):
    """3カテゴリーの推奨レースを選定（オッズ基準・シンプル版 v2.5）"""

    # 新馬・未勝利は除外
    target_df = result_df[result_df['is_maiden'] == False].copy()

    # オッズがない馬を除外
    target_df = target_df[target_df['odds_win'].notna()]

    recommended = {}
    used_race_ids = set()

    # ============================================================
    # 🔵 鉄板: AIランク1位 × オッズ1.0〜5.0倍 × 1〜3番人気
    # 目的: 堅実に的中、馬連〜2,000円
    # ============================================================
    teppan_candidates = target_df[
        (target_df['pred_rank'] == 1) &
        (target_df['popularity'] >= 1) &
        (target_df['popularity'] <= 3) &
        (target_df['odds_win'] >= 1.0) &
        (target_df['odds_win'] <= 5.0) &
        (target_df['score'] >= 50.0)
    ].copy()

    if len(teppan_candidates) == 0:
        # 条件緩和: オッズ7.0倍まで、スコア45以上
        teppan_candidates = target_df[
            (target_df['pred_rank'] == 1) &
            (target_df['popularity'] >= 1) &
            (target_df['popularity'] <= 4) &
            (target_df['odds_win'] >= 1.0) &
            (target_df['odds_win'] <= 7.0) &
            (target_df['score'] >= 45.0)
        ].copy()

    if len(teppan_candidates) > 0:
        # スコア最高のレースを選択
        best = teppan_candidates.loc[teppan_candidates['score'].idxmax()]
        race_id = best['race_id']
        race_df = result_df[result_df['race_id'] == race_id].copy()
        recommended['TEPPAN'] = create_recommendation(best, race_df, 'TEPPAN')
        used_race_ids.add(race_id)

    # ============================================================
    # 🟡 中穴: AIランク1〜3位 × オッズ7.0倍以上
    # 目的: 馬連2,000〜10,000円
    # 人気条件撤廃 → オッズのみで判定
    # ============================================================
    chuuana_candidates = target_df[
        (~target_df['race_id'].isin(used_race_ids)) &
        (target_df['pred_rank'] <= 3) &
        (target_df['odds_win'] >= 7.0) &
        (target_df['score'] >= 20.0)
    ].copy()

    if len(chuuana_candidates) == 0:
        # 条件緩和: オッズ5.0倍以上、スコア15以上
        chuuana_candidates = target_df[
            (~target_df['race_id'].isin(used_race_ids)) &
            (target_df['pred_rank'] <= 3) &
            (target_df['odds_win'] >= 5.0) &
            (target_df['score'] >= 15.0)
        ].copy()

    if len(chuuana_candidates) > 0:
        # スコア×オッズで価値が高いものを選択
        chuuana_candidates['value'] = chuuana_candidates['score'] * np.log1p(chuuana_candidates['odds_win'])
        best = chuuana_candidates.loc[chuuana_candidates['value'].idxmax()]
        race_id = best['race_id']
        race_df = result_df[result_df['race_id'] == race_id].copy()
        recommended['CHUUANA'] = create_recommendation(best, race_df, 'CHUUANA')
        used_race_ids.add(race_id)

    # ============================================================
    # 🔴 大穴: オッズ20倍以上の中でスコア最高の馬
    # 目的: 馬連10,000円以上（万馬券）
    # シンプル: オッズ20倍以上 → スコア最高を選定
    # ============================================================
    ooana_candidates = target_df[
        (~target_df['race_id'].isin(used_race_ids)) &
        (target_df['odds_win'] >= 20.0)
    ].copy()

    if len(ooana_candidates) == 0:
        # 条件緩和: オッズ15倍以上
        ooana_candidates = target_df[
            (~target_df['race_id'].isin(used_race_ids)) &
            (target_df['odds_win'] >= 15.0)
        ].copy()

    if len(ooana_candidates) > 0:
        # スコア最高の馬を選択
        best = ooana_candidates.loc[ooana_candidates['score'].idxmax()]
        race_id = best['race_id']
        race_df = result_df[result_df['race_id'] == race_id].copy()
        recommended['OOANA'] = create_recommendation(best, race_df, 'OOANA')

    return recommended


def create_recommendation(honmei_row, race_df, cat_key):
    """推奨レース情報を作成（買い目付き）"""
    config = CATEGORY_CONFIG[cat_key]
    
    # レース内の馬をAIスコア順にソート
    race_df = race_df.sort_values('pred_rank')
    
    # 印をつける馬を取得（◎○▲△☆）
    top_horses = race_df.head(7).to_dict('records')
    
    # 印の割り当て
    marks = ['◎', '○', '▲', '△', '△', '☆', '☆']
    marked_horses = []
    for i, horse in enumerate(top_horses):
        marked_horses.append({
            'mark': marks[i] if i < len(marks) else '×',
            'horse_no': int(horse['horse_no']),
            'horse_name': horse['horse_name'],
            'jockey_name': horse['jockey_name'],
            'odds_win': float(horse['odds_win']) if pd.notna(horse['odds_win']) else None,
            'popularity': int(horse['popularity']) if pd.notna(horse['popularity']) else None,
            'score': round(horse['score'], 1),
            'rank': int(horse['pred_rank']),
            'gap': int(horse['gap']) if pd.notna(horse.get('gap')) else 0
        })
    
    # 買い目生成
    bets = generate_bets(marked_horses, cat_key)
    
    return {
        'race_id': honmei_row['race_id'],
        'place_name': honmei_row['place_name'],
        'race_no': int(honmei_row['race_no']),
        'race_name': honmei_row['race_name'],
        'track_type': honmei_row['track_type'],
        'distance': int(honmei_row['distance']),
        'field_size': int(honmei_row['field_size']),
        'category': cat_key,
        'category_name': config['name'],
        'category_emoji': config['emoji'],
        'description': config['description'],
        'bet_style': config['bet_style'],
        # 本命馬情報
        'honmei': {
            'horse_no': int(honmei_row['horse_no']),
            'horse_name': honmei_row['horse_name'],
            'jockey_name': honmei_row['jockey_name'],
            'score': round(honmei_row['score'], 1),
            'odds_win': float(honmei_row['odds_win']) if pd.notna(honmei_row['odds_win']) else None,
            'popularity': int(honmei_row['popularity']) if pd.notna(honmei_row['popularity']) else None,
            'gap': int(honmei_row['gap']) if pd.notna(honmei_row.get('gap')) else 0
        },
        # 印付き馬リスト
        'marked_horses': marked_horses,
        # 買い目
        'bets': bets
    }


def generate_bets(marked_horses, cat_key):
    """カテゴリー別の買い目を生成"""
    
    if len(marked_horses) < 3:
        return {'error': '馬が不足しています'}
    
    # 馬番を取得
    h1 = marked_horses[0]['horse_no']  # ◎
    h2 = marked_horses[1]['horse_no']  # ○
    h3 = marked_horses[2]['horse_no']  # ▲
    h4 = marked_horses[3]['horse_no'] if len(marked_horses) > 3 else None  # △
    h5 = marked_horses[4]['horse_no'] if len(marked_horses) > 4 else None  # △
    h6 = marked_horses[5]['horse_no'] if len(marked_horses) > 5 else None  # ☆
    h7 = marked_horses[6]['horse_no'] if len(marked_horses) > 6 else None  # ☆
    
    bets = {}
    
    # ============================================================
    # 単勝（全カテゴリー共通: ◎1点）
    # ============================================================
    bets['tansho'] = {
        'type': '単勝',
        'points': 1,
        'bets': [f"{h1}"]
    }
    
    # ============================================================
    # 馬連
    # ============================================================
    if cat_key == 'TEPPAN':
        # 鉄板: ◎-○ 1点
        bets['umaren'] = {
            'type': '馬連',
            'points': 1,
            'bets': [f"{h1}-{h2}"]
        }
    elif cat_key == 'CHUUANA':
        # 中穴: ◎-○▲ 2点
        bets['umaren'] = {
            'type': '馬連',
            'points': 2,
            'bets': [f"{h1}-{h2}", f"{h1}-{h3}"]
        }
    else:  # OOANA
        # 大穴: ◎-○▲△ 3点
        umaren_bets = [f"{h1}-{h2}", f"{h1}-{h3}"]
        if h4:
            umaren_bets.append(f"{h1}-{h4}")
        bets['umaren'] = {
            'type': '馬連',
            'points': len(umaren_bets),
            'bets': umaren_bets
        }
    
    # ============================================================
    # 3連複
    # ============================================================
    if cat_key == 'TEPPAN':
        # 鉄板: ◎-○▲-○▲△ (6点)
        # フォーメーション: 1着◎、2着○▲、3着○▲△
        sanrenpuku_bets = []
        aite_2 = [h2, h3]
        aite_3 = [h2, h3]
        if h4:
            aite_3.append(h4)
        
        for a2 in aite_2:
            for a3 in aite_3:
                if a2 != a3:
                    combo = tuple(sorted([h1, a2, a3]))
                    bet_str = f"{combo[0]}-{combo[1]}-{combo[2]}"
                    if bet_str not in sanrenpuku_bets:
                        sanrenpuku_bets.append(bet_str)
        
        bets['sanrenpuku'] = {
            'type': '3連複',
            'points': len(sanrenpuku_bets),
            'formation': f"◎{h1} - ○▲ - ○▲△",
            'bets': sanrenpuku_bets
        }
        
    elif cat_key == 'CHUUANA':
        # 中穴: ◎-○▲△-○▲△☆ (12点程度)
        sanrenpuku_bets = []
        aite_2 = [h2, h3]
        if h4:
            aite_2.append(h4)
        aite_3 = [h2, h3]
        if h4:
            aite_3.append(h4)
        if h6:
            aite_3.append(h6)
        
        for a2 in aite_2:
            for a3 in aite_3:
                if a2 != a3:
                    combo = tuple(sorted([h1, a2, a3]))
                    bet_str = f"{combo[0]}-{combo[1]}-{combo[2]}"
                    if bet_str not in sanrenpuku_bets:
                        sanrenpuku_bets.append(bet_str)
        
        bets['sanrenpuku'] = {
            'type': '3連複',
            'points': len(sanrenpuku_bets),
            'formation': f"◎{h1} - ○▲△ - ○▲△☆",
            'bets': sanrenpuku_bets
        }
        
    else:  # OOANA
        # 大穴: 2-4-7形式 (19点)
        # 1列目: ◎○、2列目: ◎○▲△、3列目: ○▲△△☆☆★
        sanrenpuku_bets = []
        col1 = [h1, h2]
        col2 = [h1, h2, h3]
        if h4:
            col2.append(h4)
        col3 = [h2, h3]
        if h4:
            col3.append(h4)
        if h5:
            col3.append(h5)
        if h6:
            col3.append(h6)
        if h7:
            col3.append(h7)
        
        for c1 in col1:
            for c2 in col2:
                for c3 in col3:
                    if len(set([c1, c2, c3])) == 3:  # 重複なし
                        combo = tuple(sorted([c1, c2, c3]))
                        bet_str = f"{combo[0]}-{combo[1]}-{combo[2]}"
                        if bet_str not in sanrenpuku_bets:
                            sanrenpuku_bets.append(bet_str)
        
        bets['sanrenpuku'] = {
            'type': '3連複',
            'points': len(sanrenpuku_bets),
            'formation': f"◎○ - ◎○▲△ - ○▲△△☆☆",
            'bets': sanrenpuku_bets
        }
    
    # ============================================================
    # 3連単
    # ============================================================
    if cat_key == 'TEPPAN':
        # 鉄板: ◎→○▲→○▲△ (6点)
        sanrentan_bets = []
        for second in [h2, h3]:
            for third in [h2, h3]:
                if h4:
                    for t in [h2, h3, h4]:
                        if second != t and h1 != t:
                            bet_str = f"{h1}→{second}→{t}"
                            if bet_str not in sanrentan_bets:
                                sanrentan_bets.append(bet_str)
                else:
                    if second != third:
                        bet_str = f"{h1}→{second}→{third}"
                        if bet_str not in sanrentan_bets:
                            sanrentan_bets.append(bet_str)
        
        bets['sanrentan'] = {
            'type': '3連単',
            'points': len(sanrentan_bets),
            'formation': f"◎{h1} → ○▲ → ○▲△",
            'bets': sanrentan_bets
        }
        
    elif cat_key == 'CHUUANA':
        # 中穴: ◎→○▲→○▲△△ (12点程度)
        sanrentan_bets = []
        seconds = [h2, h3]
        thirds = [h2, h3]
        if h4:
            thirds.append(h4)
        if h5:
            thirds.append(h5)
        
        for second in seconds:
            for third in thirds:
                if second != third:
                    bet_str = f"{h1}→{second}→{third}"
                    if bet_str not in sanrentan_bets:
                        sanrentan_bets.append(bet_str)
        
        bets['sanrentan'] = {
            'type': '3連単',
            'points': len(sanrentan_bets),
            'formation': f"◎{h1} → ○▲ → ○▲△△",
            'bets': sanrentan_bets
        }
        
    else:  # OOANA
        # 大穴: ◎○→◎○▲→○▲△△△ (24点程度)
        sanrentan_bets = []
        firsts = [h1, h2]
        seconds = [h1, h2, h3]
        thirds = [h2, h3]
        if h4:
            thirds.append(h4)
        if h5:
            thirds.append(h5)
        if h6:
            thirds.append(h6)
        
        for first in firsts:
            for second in seconds:
                for third in thirds:
                    if len(set([first, second, third])) == 3:
                        bet_str = f"{first}→{second}→{third}"
                        if bet_str not in sanrentan_bets:
                            sanrentan_bets.append(bet_str)
        
        bets['sanrentan'] = {
            'type': '3連単',
            'points': len(sanrentan_bets),
            'formation': f"◎○ → ◎○▲ → ○▲△△△",
            'bets': sanrentan_bets
        }
    
    # 合計点数
    total_points = sum(b['points'] for b in bets.values() if isinstance(b, dict) and 'points' in b)
    bets['total_points'] = total_points
    
    return bets


def print_predictions(pred_df, recommended):
    """予測結果を表示"""

    # まず推奨レース3選を表示
    print()
    print("=" * 70)
    print("🏆 本日の推奨レース3選【人気の盲点で選定】")
    print("=" * 70)

    for cat_key in ['TEPPAN', 'CHUUANA', 'OOANA']:
        if cat_key in recommended:
            r = recommended[cat_key]
            config = CATEGORY_CONFIG[cat_key]
            honmei = r['honmei']
            bets = r['bets']
            
            print()
            print(f"{config['emoji']} 【{config['name']}】{config['description']}")
            print("-" * 60)
            print(f"📍 {r['place_name']}{r['race_no']}R {r['race_name']}")
            print(f"   {r['track_type']}{r['distance']}m ｜ {r['field_size']}頭立")
            print()
            
            # 本命馬情報
            gap_str = f"+{honmei['gap']}" if honmei['gap'] > 0 else str(honmei['gap'])
            print(f"   ◎本命: {honmei['horse_no']}番 {honmei['horse_name']} ({honmei['jockey_name']})")
            print(f"   　　　 スコア: {honmei['score']}% | オッズ: {honmei['odds_win']:.1f}倍 | {honmei['popularity']}番人気")
            print(f"   　　　 ギャップ: {gap_str} {'👀人気の盲点!' if honmei['gap'] >= 2 else ''}")
            print()
            
            # 印付き馬一覧
            print("   【予想印】")
            for h in r['marked_horses'][:5]:
                gap_mark = f"(+{h['gap']})" if h['gap'] > 0 else f"({h['gap']})" if h['gap'] < 0 else ""
                pop_str = f"{h['popularity']}人気" if h['popularity'] else "-"
                print(f"   {h['mark']} {h['horse_no']:2d} {h['horse_name'][:8]:<10} [{h['score']:>5.1f}%] {h['odds_win']:>5.1f}倍 {pop_str} {gap_mark}")
            print()
            
            # 買い目サマリー
            print(f"   【買い目】合計 {bets['total_points']}点")
            print(f"   　単勝: {', '.join(bets['tansho']['bets'])} ({bets['tansho']['points']}点)")
            print(f"   　馬連: {', '.join(bets['umaren']['bets'])} ({bets['umaren']['points']}点)")
            print(f"   　3連複: {bets['sanrenpuku']['formation']} ({bets['sanrenpuku']['points']}点)")
            print(f"   　3連単: {bets['sanrentan']['formation']} ({bets['sanrentan']['points']}点)")
            
        else:
            config = CATEGORY_CONFIG[cat_key]
            print()
            print(f"{config['emoji']} 【{config['name']}】該当レースなし")

    print()
    print("=" * 70)
    print("📋 全レース予想一覧")
    print("=" * 70)

    for place_code in sorted(pred_df['place_code'].unique()):
        place_df = pred_df[pred_df['place_code'] == place_code]
        place_name = place_df['place_name'].iloc[0]

        for race_id in sorted(place_df['race_id'].unique()):
            race_df = place_df[place_df['race_id'] == race_id].copy()
            race_df = race_df.sort_values('pred_rank')
            first = race_df.iloc[0]

            # 推奨レースかどうかチェック
            rec_mark = ""
            for cat_key, r in recommended.items():
                if r['race_id'] == race_id:
                    rec_mark = f" {CATEGORY_CONFIG[cat_key]['emoji']}推奨"
                    break

            race_type_str = ""
            if first['is_maiden']:
                race_type_str = " ⚠️新馬/未勝利"
            elif first['is_graded']:
                race_type_str = " ★重賞"

            print()
            print(f"📍 {place_name}{first['race_no']}R {first['race_name']}{race_type_str}{rec_mark}")
            print(f"   {first['track_type']}{first['distance']}m {first['field_size']}頭立")

            marks = ['◎', '○', '▲', '△', '☆']
            for i, (_, row) in enumerate(race_df.head(5).iterrows()):
                mark = marks[i] if i < len(marks) else '  '
                odds_str = f"[{row['odds_win']:.1f}倍]" if pd.notna(row['odds_win']) else ""
                pop_str = f"{int(row['popularity'])}人気" if pd.notna(row['popularity']) else ""
                gap_str = f"(+{int(row['gap'])})" if pd.notna(row.get('gap')) and row['gap'] > 0 else ""
                print(f"   {mark} {row['horse_no']:2d} {row['horse_name'][:10]:<12} [{row['score']:.1f}%] {odds_str} {pop_str} {gap_str}")


def generate_output(pred_df, recommended, has_odds):
    """予測結果をJSON出力"""
    date_str = pred_df['date'].iloc[0]
    places_data = []

    for place_code in sorted(pred_df['place_code'].unique()):
        place_df = pred_df[pred_df['place_code'] == place_code]
        place_name = place_df['place_name'].iloc[0]
        races_data = []

        for race_id in sorted(place_df['race_id'].unique()):
            race_df = place_df[place_df['race_id'] == race_id].copy()
            race_df = race_df.sort_values('pred_rank')
            first = race_df.iloc[0]

            horses_data = []
            for _, row in race_df.iterrows():
                horse = {
                    'horse_no': int(row['horse_no']),
                    'horse_name': row['horse_name'],
                    'jockey_name': row['jockey_name'],
                    'load_weight': float(row['load_weight']) if pd.notna(row['load_weight']) else 0,
                    'sex_age': f"{row['horse_sex']}{row['horse_age']}",
                    'odds_win': float(row['odds_win']) if pd.notna(row['odds_win']) else None,
                    'popularity': int(row['popularity']) if pd.notna(row['popularity']) else None,
                    'score': round(row['score'], 1),
                    'rank': int(row['pred_rank']),
                    'gap': int(row['gap']) if pd.notna(row.get('gap')) else 0
                }
                horses_data.append(horse)

            # 印の割り当て
            marks = ['◎', '○', '▲', '△', '△', '☆', '☆']
            for i, horse in enumerate(horses_data[:7]):
                horse['mark'] = marks[i] if i < len(marks) else '×'

            top3 = race_df.head(3)['horse_no'].astype(int).tolist()
            top5 = race_df.head(5)['horse_no'].astype(int).tolist()

            race_data = {
                'race_id': race_id,
                'race_no': int(first['race_no']),
                'race_name': first['race_name'],
                'race_class': first['race_class'],
                'is_graded': bool(first['is_graded']),
                'is_maiden': bool(first['is_maiden']),
                'distance': int(first['distance']),
                'track_type': first['track_type'],
                'field_size': int(first['field_size']),
                'top_score': round(first['score'], 1),
                'horses': horses_data,
                'recommended_bets': {
                    'tansho': f"{top3[0]}",
                    'umaren': f"{top3[0]}-{top3[1]}",
                    'umatan': f"{top3[0]}→{top3[1]}",
                    'wide': [f"{top3[0]}-{top3[1]}", f"{top3[0]}-{top3[2]}"],
                    'sanrenpuku': f"{top3[0]}-{top3[1]}-{top3[2]}",
                    'sanrentan': f"{top3[0]}→{top3[1]}→{top3[2]}"
                }
            }
            races_data.append(race_data)

        place_data = {
            'place_code': place_code,
            'place_name': place_name,
            'races': races_data
        }
        places_data.append(place_data)

    output = {
        'date': date_str,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'has_odds': has_odds,
        'model_version': '2.4',
        'selection_method': '人気の盲点（ギャップ指標）',
        'bet_types': ['単勝', '馬連', '3連複', '3連単'],
        'recommended_races': recommended,
        'places': places_data
    }

    return output


def main():
    parser = argparse.ArgumentParser(description='改善版予測 v2.4 - 人気の盲点ベース')
    parser.add_argument('date', nargs='?', default=None, help='日付 (YYYYMMDD)')
    args = parser.parse_args()

    print("=" * 70)
    print("🏇 Keiba Intelligence v2.4 - 人気の盲点で選ぶ3カテゴリー推奨")
    print("=" * 70)
    print("📌 選定ロジック: ギャップ = 人気順 - AIランク")
    print("   プラスが大きいほど「過小評価されている馬」= 狙い目")
    print()
    print("📌 カテゴリー別戦略:")
    print("   🔵 鉄板: AIランク1位 × 1〜2番人気 → 順当な本命")
    print("   🟡 中穴: AIランク上位 × 中位人気 × ギャップ+2以上 → 実力馬の過小評価")
    print("   🔴 大穴: AIランク上位 × 下位人気 × ギャップ+3以上 → 人気の盲点")
    print()
    print("📌 買い目: 単勝・馬連・3連複・3連単（TOP4馬券種）")
    print()

    if args.date:
        date_str = args.date.replace('-', '').replace('/', '')
    else:
        csv_files = list(ENTRY_DIR.glob("DE*.CSV"))
        if not csv_files:
            print("❌ 出馬表CSVが見つかりません")
            return
        latest = sorted(csv_files)[-1]
        date_str = '20' + latest.stem[2:]

    print(f"📅 対象日: {date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}")
    print()

    entry_df = load_entry_csv(date_str)
    if entry_df is None:
        return

    odds_df = load_odds_csv(date_str)
    has_odds = odds_df is not None and len(odds_df) > 0

    if has_odds:
        merged_df = merge_entry_and_odds(entry_df, odds_df)
    else:
        merged_df = entry_df.copy()
        merged_df['place_code'] = merged_df['場所'].map(PLACE_CSV_TO_CODE)
        merged_df['odds_win'] = None

    pred_df = convert_to_predict_format(merged_df, date_str[2:])
    hist_df = load_historical_data_from_csv()
    model, feature_cols = train_model(hist_df)
    result_df = predict_races(model, feature_cols, pred_df)

    # 推奨レース選定（人気の盲点ベース）
    recommended = select_recommended_races(result_df)

    # 結果表示
    print_predictions(result_df, recommended)


    # 予測ログ記録
    try:
        logger = PredictionLogger(model_version='3.0')
        features_df = create_features(result_df, is_prediction=True)
        track_condition = result_df['track_condition'].iloc[0] if 'track_condition' in result_df.columns else '良'
        weather = result_df['weather'].iloc[0] if 'weather' in result_df.columns else '晴'
        log_result = logger.log_predictions(result_df, features_df, [], track_condition, weather)
        print(f"📝 予測ログ記録: {log_result.get('logged', 0)}件")
    except Exception as e:
        print(f"⚠️  予測ログ記録エラー: {e}")

    # JSON保存
    output = generate_output(result_df, recommended, has_odds)

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PREDICTION_DIR / f"predictions_{date_str}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("📊 予測サマリー")
    print("=" * 70)
    print(f"  総レース数: {result_df['race_id'].nunique()}")
    print(f"  出走頭数: {len(result_df)}")
    print(f"  オッズデータ: {'あり ✅' if has_odds else 'なし ⚠️'}")
    print()
    
    # 推奨レースの買い目点数サマリー
    total_bets = 0
    for cat_key in ['TEPPAN', 'CHUUANA', 'OOANA']:
        if cat_key in recommended:
            points = recommended[cat_key]['bets']['total_points']
            total_bets += points
            print(f"  {CATEGORY_CONFIG[cat_key]['emoji']} {CATEGORY_CONFIG[cat_key]['name']}: {points}点")
    print(f"  ─────────────────")
    print(f"  📝 推奨買い目合計: {total_bets}点")
    print()
    print(f"💾 JSON保存: {json_path}")


if __name__ == '__main__':
    main()
















