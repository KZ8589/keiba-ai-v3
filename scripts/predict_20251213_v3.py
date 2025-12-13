"""
12月13日レース予測（買い目構成版）
- 中穴・大穴を狙った買い目を生成
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from itertools import permutations

RACECARD_PATH = Path('data/csv_imports/racecard/DE251213.CSV')
ODDS_PATH = Path('data/csv_imports/odds/OD251213.CSV')
CSV_PATH = Path('data/csv_imports/results/20150105_20251130all.csv')

# オッズ区分
ODDS_CATEGORY = {
    '本命': (0, 3),
    '対抗': (3, 10),
    '中穴': (10, 30),
    '大穴': (30, float('inf'))
}

FEATURE_COLS = [
    'odds_log', 'popularity', 'horse_age', 'load_weight', 'gate_no',
    'field_size', 'distance', 'popularity_ratio', 'odds_rank',
    'is_turf', 'horse_sex', 'place_code', 'rest_weeks'
]

def get_odds_category(odds):
    """オッズからカテゴリを取得"""
    if pd.isna(odds):
        return '不明'
    for cat, (low, high) in ODDS_CATEGORY.items():
        if low < odds <= high:
            return cat
    return '不明'

def create_features(df, for_prediction=False):
    features = pd.DataFrame(index=df.index)
    features['odds_log'] = np.log1p(df['odds_win'].fillna(100))
    features['popularity'] = df['popularity'].fillna(10)
    features['horse_age'] = pd.to_numeric(df['horse_age'], errors='coerce').fillna(4)
    features['load_weight'] = pd.to_numeric(df['load_weight'], errors='coerce').fillna(55)
    features['gate_no'] = pd.to_numeric(df['gate_no'], errors='coerce').fillna(4)
    features['field_size'] = pd.to_numeric(df['field_size'], errors='coerce').fillna(14)
    features['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600)
    features['popularity_ratio'] = features['popularity'] / features['field_size']
    features['odds_rank'] = df.groupby('race_id')['odds_win'].rank(method='min').fillna(10)
    features['is_turf'] = (df['track_type'] == '芝').astype(int)
    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    features['horse_sex'] = df['horse_sex'].map(sex_map).fillna(0)
    if for_prediction:
        place_code_map = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7, '09': 8, '10': 9}
        features['place_code'] = df['place_code'].map(place_code_map).fillna(5)
    else:
        features['place_code'] = pd.Categorical(df['place_code'].fillna('05')).codes
    features['rest_weeks'] = pd.to_numeric(df['rest_weeks'], errors='coerce').fillna(4).clip(0, 52)
    return features[FEATURE_COLS]

def train_model():
    print("📚 モデル訓練中...")
    df = pd.read_csv(CSV_PATH, encoding='cp932', low_memory=False)
    df['date'] = df['日付'].apply(lambda x: f"20{str(x)[:2]}-{str(x)[2:4]}-{str(x)[4:6]}" if pd.notna(x) else None)
    df = df[df['date'] <= '2025-12-12']
    
    def zen_to_han(s):
        if pd.isna(s): return np.nan
        s = str(s)
        for z, h in zip('０１２３４５６７８９', '0123456789'):
            s = s.replace(z, h)
        try: return int(s) if s.isdigit() else np.nan
        except: return np.nan
    
    df['finish_position'] = df['着順'].apply(zen_to_han)
    df = df[df['finish_position'].notna() & (df['finish_position'] > 0)]
    
    place_to_code = {'札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
                     '東京': '05', '中山': '06', '中京': '07', '京都': '08',
                     '阪神': '09', '小倉': '10'}
    df['place_code'] = df['場所'].map(place_to_code)
    df['race_id'] = df['date'] + '_' + df['place_code'] + '_' + df['Ｒ'].astype(str).str.zfill(2)
    
    df = df.rename(columns={
        '単勝オッズ': 'odds_win', '人気': 'popularity', '年齢': 'horse_age',
        '性別': 'horse_sex', '斤量': 'load_weight', '枠番': 'gate_no',
        '頭数': 'field_size', '芝・ダ': 'track_type', '距離': 'distance',
        '間隔': 'rest_weeks'
    })
    
    for col in ['odds_win', 'popularity', 'horse_age', 'load_weight', 'gate_no', 
                'field_size', 'distance', 'rest_weeks']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['track_type'] = df['track_type'].replace({'ダ': 'ダート'})
    
    features = create_features(df, for_prediction=False)
    y = (df['finish_position'] == 1).astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=500, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    print(f"  → 訓練データ: {len(df):,}件, AUC: {model.best_score['valid_0']['auc']:.4f}")
    return model

def load_racecard():
    df = pd.read_csv(RACECARD_PATH, encoding='cp932')
    df = df.rename(columns={
        '年月日': 'date_raw', '場所': 'place', 'R': 'race_no', '馬番': 'horse_no',
        'レース名': 'race_name', '芝・ダ': 'track_type', '距離': 'distance',
        '馬名': 'horse_name', '性別': 'horse_sex', '年齢': 'horse_age',
        '騎手': 'jockey_name', '斤量': 'load_weight', '調教師': 'trainer_name',
        '枠': 'gate_no', '間隔': 'rest_weeks', '前人気': 'prev_popularity',
        '前着': 'prev_finish', '頭数': 'field_size', 'レースID': 'race_id_raw'
    })
    place_to_code = {'札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
                     '東京': '05', '中山': '06', '中京': '07', '京都': '08',
                     '阪神': '09', '小倉': '10'}
    df['place_code'] = df['place'].map(place_to_code)
    df['date'] = '2025-12-13'
    df['race_id'] = df['date'] + '_' + df['place_code'] + '_' + df['race_no'].astype(str).str.zfill(2)
    df['track_type'] = df['track_type'].replace({'ダ': 'ダート'})
    return df

def load_odds():
    df = pd.read_csv(ODDS_PATH, encoding='cp932', header=None)
    odds = df[[0, 4, 7]].copy()
    odds.columns = ['race_id_raw', 'horse_no', 'odds_win']
    odds['race_id_raw'] = odds['race_id_raw'].astype(str).str.strip()
    odds['horse_no'] = pd.to_numeric(odds['horse_no'], errors='coerce')
    odds['odds_win'] = pd.to_numeric(odds['odds_win'], errors='coerce')
    return odds

def predict_races(model):
    print("\n🏇 レース予測中...")
    racecard = load_racecard()
    odds = load_odds()
    racecard['race_id_raw'] = racecard['race_id_raw'].astype(str).str.strip()
    df = racecard.merge(odds, on=['race_id_raw', 'horse_no'], how='left')
    print(f"  → 出馬表: {len(racecard)}頭, オッズ結合後: {len(df)}頭")
    
    df['popularity'] = df.groupby('race_id')['odds_win'].rank(method='min').fillna(10)
    df['odds_category'] = df['odds_win'].apply(get_odds_category)
    
    features = create_features(df, for_prediction=True)
    df['pred_prob'] = model.predict(features)
    df['score'] = df['pred_prob'] * 100
    df['pred_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first').astype(int)
    
    return df

def generate_betting_tickets(race_df):
    """買い目を生成"""
    tickets = {
        'tansho': [],      # 単勝
        'umaren': [],      # 馬連
        'sanrentan': []    # 3連単
    }
    
    # 予測順にソート
    sorted_df = race_df.sort_values('pred_rank')
    
    # 上位5頭を取得
    top5 = sorted_df.head(5)
    
    # 中穴・大穴の馬を抽出（予測上位10位以内かつ中穴/大穴）
    ana_candidates = sorted_df[
        (sorted_df['pred_rank'] <= 10) & 
        (sorted_df['odds_category'].isin(['中穴', '大穴']))
    ]
    
    # === 単勝 ===
    # 中穴・大穴で予測スコアが高い馬
    for _, row in ana_candidates.head(2).iterrows():
        tickets['tansho'].append({
            'horse_no': int(row['horse_no']),
            'horse_name': row['horse_name'],
            'odds': row['odds_win'],
            'category': row['odds_category'],
            'score': row['score']
        })
    
    # === 馬連 ===
    # 本命/対抗の軸馬と中穴・大穴の組み合わせ
    axis_horses = sorted_df[sorted_df['odds_category'].isin(['本命', '対抗'])].head(2)
    
    for _, axis in axis_horses.iterrows():
        for _, ana in ana_candidates.head(3).iterrows():
            if axis['horse_no'] != ana['horse_no']:
                pair = tuple(sorted([int(axis['horse_no']), int(ana['horse_no'])]))
                ticket = {
                    'pair': pair,
                    'axis': f"{int(axis['horse_no'])}番{axis['horse_name'][:4]}",
                    'ana': f"{int(ana['horse_no'])}番{ana['horse_name'][:4]}({ana['odds_category']})",
                    'expected_odds': axis['odds_win'] * ana['odds_win'] / 5  # 概算
                }
                if ticket['pair'] not in [t['pair'] for t in tickets['umaren']]:
                    tickets['umaren'].append(ticket)
    
    # === 3連単 ===
    # 軸馬1着固定、中穴・大穴を2-3着に
    if len(axis_horses) > 0:
        axis = axis_horses.iloc[0]
        
        # 2着候補（対抗 + 中穴上位）
        second_candidates = sorted_df[
            (sorted_df['horse_no'] != axis['horse_no']) &
            (sorted_df['pred_rank'] <= 6)
        ].head(4)
        
        # 3着候補（中穴・大穴）
        third_candidates = ana_candidates.head(4)
        
        for _, second in second_candidates.iterrows():
            for _, third in third_candidates.iterrows():
                if second['horse_no'] != third['horse_no'] and axis['horse_no'] != third['horse_no']:
                    tickets['sanrentan'].append({
                        'order': (int(axis['horse_no']), int(second['horse_no']), int(third['horse_no'])),
                        'display': f"{int(axis['horse_no'])}-{int(second['horse_no'])}-{int(third['horse_no'])}",
                        'third_category': third['odds_category']
                    })
    
    return tickets

def display_predictions(df):
    print("\n" + "="*90)
    print("📊 12月13日 レース予測結果（中穴・大穴狙い）")
    print("="*90)
    
    all_tickets = []
    
    for race_id in sorted(df['race_id'].unique()):
        race_df = df[df['race_id'] == race_id].copy()
        sorted_df = race_df.sort_values('pred_rank')
        
        place = sorted_df['place'].iloc[0]
        race_no = sorted_df['race_no'].iloc[0]
        race_name = sorted_df['race_name'].iloc[0]
        track = sorted_df['track_type'].iloc[0]
        dist = sorted_df['distance'].iloc[0]
        
        print(f"\n{'─'*90}")
        print(f"🏇 {place} {race_no}R {race_name} ({track}{dist}m)")
        print(f"{'─'*90}")
        
        # 予測結果
        print(f"{'予想':^4} | {'馬番':^4} | {'馬名':^12} | {'オッズ':^7} | {'区分':^6} | {'スコア':^7}")
        print(f"{'─'*60}")
        
        for i, (_, row) in enumerate(sorted_df.head(6).iterrows()):
            rank_mark = ['◎', '○', '▲', '△', '☆', '注'][i]
            cat = row['odds_category']
            cat_mark = {'本命': '', '対抗': '', '中穴': '★', '大穴': '★★'}.get(cat, '')
            print(f" {rank_mark}  | {int(row['horse_no']):^4} | {row['horse_name'][:6]:^12} | {row['odds_win']:>6.1f} | {cat}{cat_mark:^4} | {row['score']:>6.1f}")
        
        # 買い目生成
        tickets = generate_betting_tickets(race_df)
        
        # 中穴・大穴がある場合のみ買い目表示
        if tickets['tansho'] or tickets['umaren']:
            print(f"\n💰 買い目（中穴・大穴狙い）")
            
            if tickets['tansho']:
                print(f"  【単勝】", end='')
                for t in tickets['tansho'][:2]:
                    print(f" {t['horse_no']}番({t['category']},オッズ{t['odds']:.1f})", end='')
                print()
            
            if tickets['umaren']:
                print(f"  【馬連】", end='')
                for t in tickets['umaren'][:4]:
                    print(f" {t['pair'][0]}-{t['pair'][1]}", end='')
                print()
            
            if tickets['sanrentan']:
                print(f"  【3連単】", end='')
                for t in tickets['sanrentan'][:6]:
                    print(f" {t['display']}", end='')
                print()
            
            all_tickets.append({
                'race': f"{place}{race_no}R",
                'tickets': tickets
            })
    
    # サマリー
    print("\n" + "="*90)
    print("📋 本日の推奨買い目サマリー")
    print("="*90)
    
    print("\n【単勝（中穴・大穴）】")
    for race_info in all_tickets:
        if race_info['tickets']['tansho']:
            race = race_info['race']
            for t in race_info['tickets']['tansho']:
                print(f"  {race}: {t['horse_no']}番 {t['horse_name'][:6]} ({t['category']}, オッズ{t['odds']:.1f})")
    
    print("\n【馬連（軸×穴）】")
    for race_info in all_tickets:
        if race_info['tickets']['umaren']:
            race = race_info['race']
            pairs = [f"{t['pair'][0]}-{t['pair'][1]}" for t in race_info['tickets']['umaren'][:3]]
            print(f"  {race}: {', '.join(pairs)}")

def main():
    model = train_model()
    df = predict_races(model)
    display_predictions(df)
    
    output_path = Path('temp/predictions_20251213_v2.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 予測結果保存: {output_path}")

if __name__ == "__main__":
    main()
