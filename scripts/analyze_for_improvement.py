"""
回収率改善のための分析
- どの条件で的中/外れが多いか
- 期待値の高い買い方を探る
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np

def parse_result_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='cp932')
    results = []
    for _, row in df.iterrows():
        race_id_raw = str(row['レースID'])
        date = f"{race_id_raw[:4]}-{race_id_raw[4:6]}-{race_id_raw[6:8]}"
        place_code = race_id_raw[8:10]
        race_no = int(race_id_raw[14:16])
        race_id = f"{date}_{place_code}_{str(race_no).zfill(2)}"
        
        results.append({
            'race_id': race_id,
            'finish_1st': int(row.get('1着馬番', 0)),
            'finish_2nd': int(row.get('2着馬番', 0)),
            'finish_3rd': int(row.get('3着馬番', 0)),
            'win_payout': row.get('単配当', 0),
            'umaren_payout': row.get('馬連配当', 0),
            'sanrentan_payout': row.get('３連単配当', 0),
            'horse_1st_odds': row.get('1着単オッズ', 0),
            'horse_1st_pop': row.get('1着人気', 0),
            'field_size': row.get('頭数', 0),
        })
    return pd.DataFrame(results)

def load_predictions(date: str) -> pd.DataFrame:
    filepath = Path(f'temp/predictions_{date.replace("-", "")}_v3.csv')
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath, encoding='utf-8-sig')

def analyze():
    print("="*70)
    print("📊 回収率改善分析")
    print("="*70)
    
    all_data = []
    
    for date in ['2025-12-13', '2025-12-14']:
        date_short = date.replace('-', '')
        result_file = Path(f'data/csv_imports/results/{date_short}_result.csv')
        if not result_file.exists():
            continue
        
        results = parse_result_csv(result_file)
        predictions = load_predictions(date)
        if predictions.empty:
            continue
        
        for _, result in results.iterrows():
            race_id = result['race_id']
            pred = predictions[predictions['race_id'] == race_id]
            if pred.empty:
                continue
            
            sorted_pred = pred.sort_values('pred_rank')
            pred_1st = sorted_pred.iloc[0]
            pred_2nd = sorted_pred.iloc[1] if len(sorted_pred) > 1 else None
            
            # オッズ区分
            odds = pred_1st['odds_win']
            if odds <= 3:
                odds_cat = '鉄板(~3倍)'
            elif odds <= 5:
                odds_cat = '人気(3-5倍)'
            elif odds <= 10:
                odds_cat = '対抗(5-10倍)'
            else:
                odds_cat = '穴(10倍~)'
            
            hit_1st = int(pred_1st['horse_no']) == result['finish_1st']
            
            all_data.append({
                'race_id': race_id,
                'pred_1st_odds': odds,
                'odds_cat': odds_cat,
                'pred_score': pred_1st['score'],
                'hit_1st': hit_1st,
                'win_payout': result['win_payout'] if hit_1st else 0,
                'field_size': result['field_size'],
                'actual_1st_pop': result['horse_1st_pop'],
            })
    
    df = pd.DataFrame(all_data)
    
    # === 分析1: オッズ帯別の成績 ===
    print("\n" + "="*70)
    print("📈 分析1: 本命馬のオッズ帯別成績")
    print("="*70)
    print(f"{'オッズ帯':<15} | {'レース':>6} | {'的中':>4} | {'的中率':>7} | {'投資':>8} | {'回収':>8} | {'回収率':>7}")
    print("-"*70)
    
    for cat in ['鉄板(~3倍)', '人気(3-5倍)', '対抗(5-10倍)', '穴(10倍~)']:
        cat_df = df[df['odds_cat'] == cat]
        if len(cat_df) == 0:
            continue
        races = len(cat_df)
        hits = cat_df['hit_1st'].sum()
        hit_rate = hits / races * 100
        bet = races * 100
        payout = cat_df['win_payout'].sum()
        roi = payout / bet * 100
        print(f"{cat:<15} | {races:>6} | {hits:>4} | {hit_rate:>6.1f}% | {bet:>7,}円 | {payout:>7,}円 | {roi:>6.1f}%")
    
    # === 分析2: 予測スコア別の成績 ===
    print("\n" + "="*70)
    print("📈 分析2: 予測スコア帯別成績")
    print("="*70)
    print(f"{'スコア帯':<15} | {'レース':>6} | {'的中':>4} | {'的中率':>7} | {'投資':>8} | {'回収':>8} | {'回収率':>7}")
    print("-"*70)
    
    score_bins = [(0, 30), (30, 40), (40, 50), (50, 100)]
    for low, high in score_bins:
        score_df = df[(df['pred_score'] >= low) & (df['pred_score'] < high)]
        if len(score_df) == 0:
            continue
        races = len(score_df)
        hits = score_df['hit_1st'].sum()
        hit_rate = hits / races * 100
        bet = races * 100
        payout = score_df['win_payout'].sum()
        roi = payout / bet * 100
        print(f"スコア{low}-{high:<8} | {races:>6} | {hits:>4} | {hit_rate:>6.1f}% | {bet:>7,}円 | {payout:>7,}円 | {roi:>6.1f}%")
    
    # === 分析3: 頭数別の成績 ===
    print("\n" + "="*70)
    print("📈 分析3: 出走頭数別成績")
    print("="*70)
    print(f"{'頭数':<15} | {'レース':>6} | {'的中':>4} | {'的中率':>7} | {'投資':>8} | {'回収':>8} | {'回収率':>7}")
    print("-"*70)
    
    size_bins = [(0, 10, '少頭数(~9頭)'), (10, 14, '標準(10-13頭)'), (14, 20, '多頭数(14頭~)')]
    for low, high, label in size_bins:
        size_df = df[(df['field_size'] >= low) & (df['field_size'] < high)]
        if len(size_df) == 0:
            continue
        races = len(size_df)
        hits = size_df['hit_1st'].sum()
        hit_rate = hits / races * 100
        bet = races * 100
        payout = size_df['win_payout'].sum()
        roi = payout / bet * 100
        print(f"{label:<15} | {races:>6} | {hits:>4} | {hit_rate:>6.1f}% | {bet:>7,}円 | {payout:>7,}円 | {roi:>6.1f}%")
    
    # === 分析4: 実際の1着人気別 ===
    print("\n" + "="*70)
    print("📈 分析4: 実際の1着馬の人気別（荒れ具合）")
    print("="*70)
    print(f"{'1着人気':<15} | {'レース':>6} | {'割合':>7}")
    print("-"*40)
    
    for pop in [1, 2, 3]:
        pop_df = df[df['actual_1st_pop'] == pop]
        races = len(pop_df)
        rate = races / len(df) * 100
        print(f"{pop}番人気が1着    | {races:>6} | {rate:>6.1f}%")
    
    pop_4plus = df[df['actual_1st_pop'] >= 4]
    print(f"4番人気以下が1着 | {len(pop_4plus):>6} | {len(pop_4plus)/len(df)*100:>6.1f}%")
    
    # === 改善提案 ===
    print("\n" + "="*70)
    print("💡 改善提案")
    print("="*70)
    
    # 最も回収率が高い条件を探す
    best_roi = 0
    best_condition = ""
    
    for cat in ['鉄板(~3倍)', '人気(3-5倍)', '対抗(5-10倍)', '穴(10倍~)']:
        cat_df = df[df['odds_cat'] == cat]
        if len(cat_df) > 0:
            roi = cat_df['win_payout'].sum() / (len(cat_df) * 100) * 100
            if roi > best_roi:
                best_roi = roi
                best_condition = f"オッズ帯={cat}"
    
    for low, high in score_bins:
        score_df = df[(df['pred_score'] >= low) & (df['pred_score'] < high)]
        if len(score_df) > 0:
            roi = score_df['win_payout'].sum() / (len(score_df) * 100) * 100
            if roi > best_roi:
                best_roi = roi
                best_condition = f"スコア{low}-{high}"
    
    print(f"\n1. 最も回収率が高い条件: {best_condition} (回収率: {best_roi:.1f}%)")
    
    print("""
2. 推奨戦略:
   ・全レース一律購入 → 条件を絞って購入
   ・期待値 = 予測確率 × オッズ > 1.0 のみ購入
   ・3連単は1点 → フォーメーション（6点程度）に変更
   
3. 追加すべき特徴量:
   ・騎手の勝率・回収率
   ・前走着順・前走オッズ
   ・馬場適性（重馬場得意など）
""")

if __name__ == "__main__":
    analyze()
