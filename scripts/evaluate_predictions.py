"""
12月13日・14日のデータ処理
- 結果CSVをDBに取り込み
- 予測→答え合わせ→分析→学習
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_PATH = Path('data/keiba.db')

# 馬場状態コード変換
TRACK_CONDITION_MAP = {
    '1': '良', '2': '稍重', '3': '重', '4': '不良',
    1: '良', 2: '稍重', 3: '重', 4: '不良'
}

def parse_result_csv(filepath: Path) -> pd.DataFrame:
    """結果CSVを解析"""
    df = pd.read_csv(filepath, encoding='cp932')
    
    results = []
    for _, row in df.iterrows():
        race_id_raw = str(row['レースID'])
        
        # レースID解析: 2025121309050307 → 2025-12-13_09_03
        # 形式: YYYYMMDDPPRRCCNN (PP=場所, RR=回, CC=日, NN=R番)
        date = f"{race_id_raw[:4]}-{race_id_raw[4:6]}-{race_id_raw[6:8]}"
        place_code = race_id_raw[8:10]
        race_no = int(race_id_raw[14:16])
        
        race_id = f"{date}_{place_code}_{str(race_no).zfill(2)}"
        
        # 馬場状態
        track_cond_code = row.get('馬場状態コード', 1)
        track_condition = TRACK_CONDITION_MAP.get(track_cond_code, '良')
        
        results.append({
            'race_id': race_id,
            'date': date,
            'place_code': place_code,
            'race_no': race_no,
            'race_name': row.get('レース名', ''),
            'distance': row.get('距離', 0),
            'track_type': '芝' if row.get('芝・ダ', 0) == 0 else 'ダート',
            'field_size': row.get('頭数', 0),
            'track_condition': track_condition,
            'finish_1st': int(row.get('1着馬番', 0)),
            'finish_2nd': int(row.get('2着馬番', 0)),
            'finish_3rd': int(row.get('3着馬番', 0)),
            'win_payout': row.get('単配当', 0),
            'umaren_payout': row.get('馬連配当', 0),
            'sanrentan_payout': row.get('３連単配当', 0),
            'horse_1st': row.get('1着馬名', ''),
            'horse_1st_odds': row.get('1着単オッズ', 0),
            'horse_1st_pop': row.get('1着人気', 0),
        })
    
    return pd.DataFrame(results)

def load_predictions(date: str) -> pd.DataFrame:
    """指定日の予測結果を読み込み"""
    filepath = Path(f'temp/predictions_{date.replace("-", "")}_v3.csv')
    if not filepath.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df

def evaluate_predictions(predictions: pd.DataFrame, results: pd.DataFrame) -> dict:
    """予測と結果を比較して評価"""
    if predictions.empty or results.empty:
        return {'total': 0, 'hit_1st': 0, 'hit_top3': 0}
    
    stats = {
        'total': 0,
        'hit_1st': 0,
        'hit_top3': 0,
        'details': []
    }
    
    for _, result in results.iterrows():
        race_id = result['race_id']
        
        # 該当レースの予測を取得
        pred = predictions[predictions['race_id'] == race_id]
        if pred.empty:
            continue
        
        # 予測1位の馬番
        pred_1st = pred.sort_values('pred_rank').iloc[0]
        pred_top3 = pred.sort_values('pred_rank').head(3)['horse_no'].tolist()
        
        actual_1st = result['finish_1st']
        actual_2nd = result['finish_2nd']
        actual_3rd = result['finish_3rd']
        
        stats['total'] += 1
        
        hit_1st = int(pred_1st['horse_no']) == actual_1st
        hit_top3 = actual_1st in [int(x) for x in pred_top3]
        
        if hit_1st:
            stats['hit_1st'] += 1
        if hit_top3:
            stats['hit_top3'] += 1
        
        stats['details'].append({
            'race_id': race_id,
            'pred_1st': int(pred_1st['horse_no']),
            'pred_1st_name': pred_1st['horse_name'],
            'pred_1st_odds': pred_1st['odds_win'],
            'actual_1st': actual_1st,
            'actual_1st_name': result['horse_1st'],
            'actual_1st_odds': result['horse_1st_odds'],
            'hit_1st': hit_1st,
            'hit_top3': hit_top3,
            'win_payout': result['win_payout'] if hit_1st else 0,
        })
    
    return stats

def main():
    print("="*70)
    print("📊 12月13日・14日 予測評価")
    print("="*70)
    
    for date in ['2025-12-13', '2025-12-14']:
        date_short = date.replace('-', '')
        result_file = Path(f'data/csv_imports/results/{date_short}_result.csv')
        
        if not result_file.exists():
            print(f"\n⚠️ {date} の結果ファイルがありません")
            continue
        
        print(f"\n{'='*70}")
        print(f"📅 {date}")
        print(f"{'='*70}")
        
        # 結果読み込み
        results = parse_result_csv(result_file)
        print(f"  結果: {len(results)}レース")
        
        # 予測読み込み
        predictions = load_predictions(date)
        if predictions.empty:
            print(f"  ⚠️ 予測ファイルがありません")
            
            # 予測を実行（既存のモデルを使用）
            print(f"  → 予測を実行します...")
            continue
        
        print(f"  予測: {len(predictions['race_id'].unique())}レース")
        
        # 評価
        stats = evaluate_predictions(predictions, results)
        
        print(f"\n📈 的中率:")
        print(f"  1着的中: {stats['hit_1st']}/{stats['total']} ({stats['hit_1st']/stats['total']*100:.1f}%)")
        print(f"  Top3的中: {stats['hit_top3']}/{stats['total']} ({stats['hit_top3']/stats['total']*100:.1f}%)")
        
        # 詳細表示（的中したレース）
        hits = [d for d in stats['details'] if d['hit_1st']]
        if hits:
            print(f"\n✅ 的中レース:")
            for h in hits:
                print(f"  {h['race_id']}: {h['pred_1st']}番{h['pred_1st_name'][:6]} → 配当{h['win_payout']}円")
        
        # 外れた人気馬（オッズ3倍以下で外れ）
        misses = [d for d in stats['details'] if not d['hit_1st'] and d['pred_1st_odds'] <= 3.0]
        if misses:
            print(f"\n❌ 人気馬が外れたレース:")
            for m in misses[:5]:
                print(f"  {m['race_id']}: 予測{m['pred_1st']}番({m['pred_1st_odds']:.1f}倍) → 実際{m['actual_1st']}番{m['actual_1st_name'][:6]}({m['actual_1st_odds']:.1f}倍)")

if __name__ == "__main__":
    main()
