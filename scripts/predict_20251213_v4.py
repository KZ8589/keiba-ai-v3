"""
12月13日レース予測（買い目形式改善版）
- 鉄板・中穴・大穴に分けて表示
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

RACECARD_PATH = Path('data/csv_imports/racecard/DE251213.CSV')
ODDS_PATH = Path('data/csv_imports/odds/OD251213.CSV')
CSV_PATH = Path('data/csv_imports/results/20150105_20251130all.csv')

# オッズ区分
ODDS_CATEGORY = {
    '鉄板': (0, 10),      # 本命+対抗
    '中穴': (10, 30),
    '大穴': (30, float('inf'))
}

FEATURE_COLS = [
    'odds_log', 'popularity', 'horse_age', 'load_weight', 'gate_no',
    'field_size', 'distance', 'popularity_ratio', 'odds_rank',
    'is_turf', 'horse_sex', 'place_code', 'rest_weeks'
]

def get_odds_category(odds):
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

def display_predictions(df):
    print("\n" + "="*70)
    print("📊 12月13日 レース予測結果")
    print("="*70)
    
    for race_id in sorted(df['race_id'].unique()):
        race_df = df[df['race_id'] == race_id].copy()
        sorted_df = race_df.sort_values('pred_rank')
        
        place = sorted_df['place'].iloc[0]
        race_no = sorted_df['race_no'].iloc[0]
        race_name = sorted_df['race_name'].iloc[0]
        track = sorted_df['track_type'].iloc[0]
        dist = sorted_df['distance'].iloc[0]
        
        print(f"\n{'='*70}")
        print(f"🏇 {place} {race_no}R {race_name} ({track}{dist}m)")
        print(f"{'='*70}")
        
        # 予測上位をカテゴリ別に分類
        top_horses = sorted_df.head(10)
        
        teppan = top_horses[top_horses['odds_category'] == '鉄板'].head(3)
        chuuana = top_horses[top_horses['odds_category'] == '中穴'].head(3)
        oana = top_horses[top_horses['odds_category'] == '大穴'].head(2)
        
        # ========== 単勝 ==========
        print("\n【単勝】")
        
        print("  ◆鉄板")
        if len(teppan) > 0:
            for _, row in teppan.head(2).iterrows():
                print(f"    {int(row['horse_no']):>2}番 {row['horse_name'][:8]} ({row['odds_win']:.1f}倍)")
        else:
            print("    ー")
        
        print("  ◆中穴")
        if len(chuuana) > 0:
            for _, row in chuuana.head(2).iterrows():
                print(f"    {int(row['horse_no']):>2}番 {row['horse_name'][:8]} ({row['odds_win']:.1f}倍)")
        else:
            print("    ー")
        
        print("  ◆大穴")
        if len(oana) > 0:
            for _, row in oana.head(2).iterrows():
                print(f"    {int(row['horse_no']):>2}番 {row['horse_name'][:8]} ({row['odds_win']:.1f}倍)")
        else:
            print("    ー")
        
        # ========== 馬連 ==========
        print("\n【馬連】")
        
        # 鉄板馬連：上位2頭の組み合わせ
        print("  ◆鉄板")
        if len(teppan) >= 2:
            h1, h2 = int(teppan.iloc[0]['horse_no']), int(teppan.iloc[1]['horse_no'])
            pair = f"{min(h1,h2)}-{max(h1,h2)}"
            est_odds = teppan.iloc[0]['odds_win'] * teppan.iloc[1]['odds_win'] / 3
            print(f"    {pair} ({est_odds:.1f}倍程度)")
        else:
            print("    ー")
        
        # 中穴馬連：鉄板×中穴
        print("  ◆中穴")
        if len(teppan) > 0 and len(chuuana) > 0:
            for _, ana in chuuana.head(2).iterrows():
                axis = teppan.iloc[0]
                h1, h2 = int(axis['horse_no']), int(ana['horse_no'])
                pair = f"{min(h1,h2)}-{max(h1,h2)}"
                est_odds = axis['odds_win'] * ana['odds_win'] / 3
                print(f"    {pair} ({est_odds:.1f}倍程度)")
        else:
            print("    ー")
        
        # 大穴馬連：鉄板×大穴
        print("  ◆大穴")
        if len(teppan) > 0 and len(oana) > 0:
            for _, ana in oana.head(2).iterrows():
                axis = teppan.iloc[0]
                h1, h2 = int(axis['horse_no']), int(ana['horse_no'])
                pair = f"{min(h1,h2)}-{max(h1,h2)}"
                est_odds = axis['odds_win'] * ana['odds_win'] / 3
                print(f"    {pair} ({est_odds:.1f}倍程度)")
        else:
            print("    ー")
        
        # ========== 3連単 ==========
        print("\n【3連単】")
        
        # 鉄板3連単：上位3頭
        print("  ◆鉄板")
        if len(teppan) >= 2:
            top3 = sorted_df.head(3)
            h1, h2, h3 = int(top3.iloc[0]['horse_no']), int(top3.iloc[1]['horse_no']), int(top3.iloc[2]['horse_no'])
            combo = f"{h1}-{h2}-{h3}"
            est_odds = top3.iloc[0]['odds_win'] * top3.iloc[1]['odds_win'] * top3.iloc[2]['odds_win'] / 5
            print(f"    {combo} ({est_odds:.1f}倍程度)")
            combo2 = f"{h1}-{h3}-{h2}"
            print(f"    {combo2} ({est_odds:.1f}倍程度)")
        else:
            print("    ー")
        
        # 中穴3連単：1着鉄板、3着に中穴
        print("  ◆中穴")
        if len(teppan) >= 2 and len(chuuana) > 0:
            axis = teppan.iloc[0]
            second = teppan.iloc[1] if len(teppan) > 1 else sorted_df.iloc[1]
            for _, third in chuuana.head(2).iterrows():
                h1, h2, h3 = int(axis['horse_no']), int(second['horse_no']), int(third['horse_no'])
                if h1 != h3 and h2 != h3:
                    combo = f"{h1}-{h2}-{h3}"
                    est_odds = axis['odds_win'] * second['odds_win'] * third['odds_win'] / 5
                    print(f"    {combo} ({est_odds:.1f}倍程度)")
        else:
            print("    ー")
        
        # 大穴3連単：1着鉄板、3着に大穴
        print("  ◆大穴")
        if len(teppan) >= 1 and len(oana) > 0:
            axis = teppan.iloc[0]
            second = teppan.iloc[1] if len(teppan) > 1 else sorted_df.iloc[1]
            for _, third in oana.head(2).iterrows():
                h1, h2, h3 = int(axis['horse_no']), int(second['horse_no']), int(third['horse_no'])
                if h1 != h3 and h2 != h3:
                    combo = f"{h1}-{h2}-{h3}"
                    est_odds = axis['odds_win'] * second['odds_win'] * third['odds_win'] / 5
                    print(f"    {combo} ({est_odds:.1f}倍程度)")
        else:
            print("    ー")

def main():
    model = train_model()
    df = predict_races(model)
    display_predictions(df)
    
    output_path = Path('temp/predictions_20251213_v3.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 予測結果保存: {output_path}")

if __name__ == "__main__":
    main()
