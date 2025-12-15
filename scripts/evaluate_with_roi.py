"""
予測評価スクリプト - 3連単対応版
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np

TRACK_CONDITION_MAP = {
    '1': '良', '2': '稍重', '3': '重', '4': '不良',
    1: '良', 2: '稍重', 3: '重', 4: '不良'
}

def parse_result_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='cp932')
    
    results = []
    for _, row in df.iterrows():
        race_id_raw = str(row['レースID'])
        date = f"{race_id_raw[:4]}-{race_id_raw[4:6]}-{race_id_raw[6:8]}"
        place_code = race_id_raw[8:10]
        race_no = int(race_id_raw[14:16])
        race_id = f"{date}_{place_code}_{str(race_no).zfill(2)}"
        
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
    filepath = Path(f'temp/predictions_{date.replace("-", "")}_v3.csv')
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath, encoding='utf-8-sig')

def evaluate_predictions(predictions: pd.DataFrame, results: pd.DataFrame, bet_unit: int = 100) -> dict:
    if predictions.empty or results.empty:
        return {'total': 0}
    
    stats = {
        'total': 0,
        'hit_1st': 0,
        'hit_top3': 0,
        'tansho_bet': 0,
        'tansho_payout': 0,
        'umaren_bet': 0,
        'umaren_payout': 0,
        'sanrentan_bet': 0,
        'sanrentan_payout': 0,
        'details': []
    }
    
    for _, result in results.iterrows():
        race_id = result['race_id']
        pred = predictions[predictions['race_id'] == race_id]
        if pred.empty:
            continue
        
        sorted_pred = pred.sort_values('pred_rank')
        pred_1st = sorted_pred.iloc[0]
        pred_2nd = sorted_pred.iloc[1] if len(sorted_pred) > 1 else None
        pred_3rd = sorted_pred.iloc[2] if len(sorted_pred) > 2 else None
        pred_top3 = sorted_pred.head(3)['horse_no'].tolist()
        
        actual_1st = result['finish_1st']
        actual_2nd = result['finish_2nd']
        actual_3rd = result['finish_3rd']
        
        stats['total'] += 1
        
        # 単勝判定
        hit_1st = int(pred_1st['horse_no']) == actual_1st
        hit_top3 = actual_1st in [int(x) for x in pred_top3]
        
        stats['tansho_bet'] += bet_unit
        if hit_1st:
            stats['hit_1st'] += 1
            stats['tansho_payout'] += result['win_payout']
        if hit_top3:
            stats['hit_top3'] += 1
        
        # 馬連判定
        umaren_hit = False
        if pred_2nd is not None:
            stats['umaren_bet'] += bet_unit
            pred_pair = set([int(pred_1st['horse_no']), int(pred_2nd['horse_no'])])
            actual_pair = set([actual_1st, actual_2nd])
            if pred_pair == actual_pair:
                stats['umaren_payout'] += result['umaren_payout']
                umaren_hit = True
        
        # 3連単判定（予測1-2-3と実際1-2-3が完全一致）
        sanrentan_hit = False
        sanrentan_payout = 0
        if pred_2nd is not None and pred_3rd is not None:
            stats['sanrentan_bet'] += bet_unit
            pred_order = (int(pred_1st['horse_no']), int(pred_2nd['horse_no']), int(pred_3rd['horse_no']))
            actual_order = (actual_1st, actual_2nd, actual_3rd)
            if pred_order == actual_order:
                stats['sanrentan_payout'] += result['sanrentan_payout']
                sanrentan_hit = True
                sanrentan_payout = result['sanrentan_payout']
        
        stats['details'].append({
            'race_id': race_id,
            'pred_1st': int(pred_1st['horse_no']),
            'pred_2nd': int(pred_2nd['horse_no']) if pred_2nd is not None else 0,
            'pred_3rd': int(pred_3rd['horse_no']) if pred_3rd is not None else 0,
            'pred_1st_name': pred_1st['horse_name'][:8],
            'pred_1st_odds': pred_1st['odds_win'],
            'actual_1st': actual_1st,
            'actual_2nd': actual_2nd,
            'actual_3rd': actual_3rd,
            'actual_1st_name': result['horse_1st'][:8] if result['horse_1st'] else '',
            'actual_1st_odds': result['horse_1st_odds'],
            'hit_1st': hit_1st,
            'hit_top3': hit_top3,
            'win_payout': result['win_payout'] if hit_1st else 0,
            'umaren_hit': umaren_hit,
            'umaren_payout': result['umaren_payout'] if umaren_hit else 0,
            'sanrentan_hit': sanrentan_hit,
            'sanrentan_payout': sanrentan_payout,
        })
    
    return stats

def display_results(date: str, stats: dict):
    if stats['total'] == 0:
        print(f"  ⚠️ データなし")
        return
    
    print(f"\n📈 的中率:")
    print(f"  1着的中: {stats['hit_1st']}/{stats['total']} ({stats['hit_1st']/stats['total']*100:.1f}%)")
    print(f"  Top3的中: {stats['hit_top3']}/{stats['total']} ({stats['hit_top3']/stats['total']*100:.1f}%)")
    
    print(f"\n💰 回収率:")
    
    # 単勝
    tansho_rate = stats['tansho_payout'] / stats['tansho_bet'] * 100 if stats['tansho_bet'] > 0 else 0
    print(f"  【単勝】  投資: {stats['tansho_bet']:,}円 → 回収: {stats['tansho_payout']:,}円 ({tansho_rate:.1f}%)")
    
    # 馬連
    umaren_rate = stats['umaren_payout'] / stats['umaren_bet'] * 100 if stats['umaren_bet'] > 0 else 0
    umaren_hits = len([d for d in stats['details'] if d['umaren_hit']])
    print(f"  【馬連】  投資: {stats['umaren_bet']:,}円 → 回収: {stats['umaren_payout']:,}円 ({umaren_rate:.1f}%) [的中{umaren_hits}件]")
    
    # 3連単
    sanrentan_rate = stats['sanrentan_payout'] / stats['sanrentan_bet'] * 100 if stats['sanrentan_bet'] > 0 else 0
    sanrentan_hits = len([d for d in stats['details'] if d['sanrentan_hit']])
    print(f"  【3連単】 投資: {stats['sanrentan_bet']:,}円 → 回収: {stats['sanrentan_payout']:,}円 ({sanrentan_rate:.1f}%) [的中{sanrentan_hits}件]")
    
    # 単勝的中レース
    hits = [d for d in stats['details'] if d['hit_1st']]
    if hits:
        print(f"\n✅ 単勝的中レース:")
        for h in hits:
            print(f"  {h['race_id']}: {h['pred_1st']}番{h['pred_1st_name']} → {h['win_payout']}円")
    
    # 馬連的中
    umaren_hits_detail = [d for d in stats['details'] if d['umaren_hit']]
    if umaren_hits_detail:
        print(f"\n✅ 馬連的中レース:")
        for h in umaren_hits_detail:
            print(f"  {h['race_id']}: {h['pred_1st']}-{h['pred_2nd']} → {h['umaren_payout']}円")
    
    # 3連単的中
    sanrentan_hits_detail = [d for d in stats['details'] if d['sanrentan_hit']]
    if sanrentan_hits_detail:
        print(f"\n✅ 3連単的中レース:")
        for h in sanrentan_hits_detail:
            print(f"  {h['race_id']}: {h['pred_1st']}-{h['pred_2nd']}-{h['pred_3rd']} → {h['sanrentan_payout']}円")
    
    # 外れた人気馬
    misses = [d for d in stats['details'] if not d['hit_1st'] and d['pred_1st_odds'] <= 3.0]
    if misses:
        print(f"\n❌ 人気馬が外れたレース:")
        for m in misses[:5]:
            print(f"  {m['race_id']}: 予測{m['pred_1st']}番({m['pred_1st_odds']:.1f}倍) → 実際{m['actual_1st']}番{m['actual_1st_name']}({m['actual_1st_odds']:.1f}倍)")

def main():
    print("="*70)
    print("📊 12月13日・14日 予測評価（回収率対応）")
    print("="*70)
    
    total_stats = {
        'total': 0, 'hit_1st': 0, 'hit_top3': 0,
        'tansho_bet': 0, 'tansho_payout': 0,
        'umaren_bet': 0, 'umaren_payout': 0,
        'sanrentan_bet': 0, 'sanrentan_payout': 0
    }
    
    for date in ['2025-12-13', '2025-12-14']:
        date_short = date.replace('-', '')
        result_file = Path(f'data/csv_imports/results/{date_short}_result.csv')
        
        if not result_file.exists():
            print(f"\n⚠️ {date} の結果ファイルがありません")
            continue
        
        print(f"\n{'='*70}")
        print(f"📅 {date}")
        print(f"{'='*70}")
        
        results = parse_result_csv(result_file)
        predictions = load_predictions(date)
        
        if predictions.empty:
            print(f"  ⚠️ 予測ファイルがありません")
            continue
        
        print(f"  結果: {len(results)}レース, 予測: {len(predictions['race_id'].unique())}レース")
        
        stats = evaluate_predictions(predictions, results)
        display_results(date, stats)
        
        for key in ['total', 'hit_1st', 'hit_top3', 'tansho_bet', 'tansho_payout', 
                    'umaren_bet', 'umaren_payout', 'sanrentan_bet', 'sanrentan_payout']:
            total_stats[key] += stats.get(key, 0)
    
    # 合計
    print(f"\n{'='*70}")
    print(f"📊 合計（2日間）")
    print(f"{'='*70}")
    
    print(f"\n📈 的中率:")
    print(f"  1着的中: {total_stats['hit_1st']}/{total_stats['total']} ({total_stats['hit_1st']/total_stats['total']*100:.1f}%)")
    print(f"  Top3的中: {total_stats['hit_top3']}/{total_stats['total']} ({total_stats['hit_top3']/total_stats['total']*100:.1f}%)")
    
    print(f"\n💰 回収率:")
    tansho_rate = total_stats['tansho_payout'] / total_stats['tansho_bet'] * 100 if total_stats['tansho_bet'] > 0 else 0
    umaren_rate = total_stats['umaren_payout'] / total_stats['umaren_bet'] * 100 if total_stats['umaren_bet'] > 0 else 0
    sanrentan_rate = total_stats['sanrentan_payout'] / total_stats['sanrentan_bet'] * 100 if total_stats['sanrentan_bet'] > 0 else 0
    print(f"  【単勝】  投資: {total_stats['tansho_bet']:,}円 → 回収: {total_stats['tansho_payout']:,}円 ({tansho_rate:.1f}%)")
    print(f"  【馬連】  投資: {total_stats['umaren_bet']:,}円 → 回収: {total_stats['umaren_payout']:,}円 ({umaren_rate:.1f}%)")
    print(f"  【3連単】 投資: {total_stats['sanrentan_bet']:,}円 → 回収: {total_stats['sanrentan_payout']:,}円 ({sanrentan_rate:.1f}%)")

if __name__ == "__main__":
    main()
