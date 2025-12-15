"""
3連単の詳細分析と改善戦略
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np
from itertools import permutations

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
            'sanrentan_payout': row.get('３連単配当', 0),
            'field_size': row.get('頭数', 0),
        })
    return pd.DataFrame(results)

def load_predictions(date: str) -> pd.DataFrame:
    filepath = Path(f'temp/predictions_{date.replace("-", "")}_v3.csv')
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath, encoding='utf-8-sig')

def analyze_sanrentan():
    print("="*70)
    print("📊 3連単 詳細分析")
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
            top5 = sorted_pred.head(5)
            top5_nos = [int(x) for x in top5['horse_no'].tolist()]
            
            actual = (result['finish_1st'], result['finish_2nd'], result['finish_3rd'])
            
            # 予測Top5に実際の1-2-3着が含まれているか
            actual_in_top5 = [a in top5_nos for a in actual]
            
            all_data.append({
                'race_id': race_id,
                'pred_top5': top5_nos,
                'actual_1st': actual[0],
                'actual_2nd': actual[1],
                'actual_3rd': actual[2],
                '1st_in_top5': actual_in_top5[0],
                '2nd_in_top5': actual_in_top5[1],
                '3rd_in_top5': actual_in_top5[2],
                'all_in_top5': all(actual_in_top5),
                'payout': result['sanrentan_payout'],
            })
    
    df = pd.DataFrame(all_data)
    
    # === 分析1: 予測Top5に何着が含まれているか ===
    print("\n" + "="*70)
    print("📈 分析1: 予測Top5に実際の着順馬が含まれる割合")
    print("="*70)
    
    print(f"  1着馬がTop5に含まれる: {df['1st_in_top5'].sum()}/{len(df)} ({df['1st_in_top5'].mean()*100:.1f}%)")
    print(f"  2着馬がTop5に含まれる: {df['2nd_in_top5'].sum()}/{len(df)} ({df['2nd_in_top5'].mean()*100:.1f}%)")
    print(f"  3着馬がTop5に含まれる: {df['3rd_in_top5'].sum()}/{len(df)} ({df['3rd_in_top5'].mean()*100:.1f}%)")
    print(f"  1-2-3着全てTop5に含まれる: {df['all_in_top5'].sum()}/{len(df)} ({df['all_in_top5'].mean()*100:.1f}%)")
    
    # === 分析2: フォーメーション買いのシミュレーション ===
    print("\n" + "="*70)
    print("📈 分析2: フォーメーション買いシミュレーション")
    print("="*70)
    
    formations = [
        ('1点買い(1-2-3)', lambda top5: [(top5[0], top5[1], top5[2])]),
        ('3点(1着固定)', lambda top5: [(top5[0], top5[1], top5[2]), 
                                        (top5[0], top5[2], top5[1]),
                                        (top5[0], top5[1], top5[3])]),
        ('6点(1着固定、2-3着2-5)', lambda top5: list(permutations([top5[0]], 1))[0:1] and
                                        [(top5[0], a, b) for a, b in permutations(top5[1:5], 2)][:6]),
        ('12点(1着1-2、2-3着2-5)', lambda top5: 
            [(top5[0], a, b) for a, b in permutations(top5[1:5], 2)][:6] +
            [(top5[1], a, b) for a, b in permutations([top5[0]] + top5[2:5], 2)][:6]),
        ('Top5ボックス(60点)', lambda top5: list(permutations(top5, 3))),
    ]
    
    print(f"{'買い方':<25} | {'点数':>4} | {'的中':>4} | {'投資':>10} | {'回収':>10} | {'回収率':>8}")
    print("-"*70)
    
    for name, formation_func in formations:
        total_bet = 0
        total_payout = 0
        hits = 0
        
        for _, row in df.iterrows():
            top5 = row['pred_top5']
            actual = (row['actual_1st'], row['actual_2nd'], row['actual_3rd'])
            
            tickets = formation_func(top5)
            bet = len(tickets) * 100
            total_bet += bet
            
            if actual in tickets:
                total_payout += row['payout']
                hits += 1
        
        roi = total_payout / total_bet * 100 if total_bet > 0 else 0
        avg_tickets = total_bet / len(df) / 100
        print(f"{name:<25} | {avg_tickets:>4.0f} | {hits:>4} | {total_bet:>9,}円 | {total_payout:>9,}円 | {roi:>7.1f}%")
    
    # === 分析3: 期待値ベースの買い目 ===
    print("\n" + "="*70)
    print("📈 分析3: 条件付き購入（期待値フィルター）")
    print("="*70)
    
    # 1-2-3着全てTop5に含まれるレースのみ購入した場合
    filtered = df[df['all_in_top5']]
    if len(filtered) > 0:
        # Top5ボックスで購入
        total_bet = len(filtered) * 60 * 100  # 60点
        total_payout = filtered['payout'].sum()
        roi = total_payout / total_bet * 100
        print(f"  1-2-3着がTop5に含まれるレース: {len(filtered)}レース")
        print(f"  Top5ボックス(60点)で購入した場合:")
        print(f"    投資: {total_bet:,}円 → 回収: {total_payout:,}円 ({roi:.1f}%)")
    
    # === 提案 ===
    print("\n" + "="*70)
    print("💡 3連単改善提案")
    print("="*70)
    print("""
【現状の問題点】
  ・1点買い(予測1-2-3)では的中率が低すぎる
  ・予測Top5に1-2-3着が全て含まれる確率は約30%程度
  
【改善戦略】
  1. フォーメーション買い
     → 1着を固定、2-3着を流す（6-12点程度）
     
  2. 荒れそうなレースを見極める
     → 頭数が多い、オッズが拮抗しているレース
     → 高配当が期待できるレースに絞る
     
  3. 期待値フィルター
     → 予測スコアが高いレースのみ購入
     → オッズ×的中確率 > 1.0 のみ購入
     
  4. 穴馬を積極的に組み込む
     → 予測4-6位の中穴馬を3着に入れる
""")

if __name__ == "__main__":
    analyze_sanrentan()
