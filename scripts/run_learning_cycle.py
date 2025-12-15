"""
自己学習サイクル実行 - パターン抽出・検証・保存
過去データ（2015-2025）からパターンを学習
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np
import json
from scipy import stats
from learning.config import FEATURE_BINS, CATEGORICAL_FEATURES, VALIDATION_CONFIG

CSV_PATH = Path('data/csv_imports/results/20150105_20251130all.csv')
PATTERN_PATH = Path('data/patterns')
PATTERN_PATH.mkdir(parents=True, exist_ok=True)

def load_historical_data():
    """過去データを読み込み"""
    print("📚 過去データ読み込み中...")
    df = pd.read_csv(CSV_PATH, encoding='cp932', low_memory=False)
    
    # 日付変換
    df['date'] = df['日付'].apply(lambda x: f"20{str(x)[:2]}-{str(x)[2:4]}-{str(x)[4:6]}" if pd.notna(x) else None)
    
    # 着順変換
    def zen_to_han(s):
        if pd.isna(s): return np.nan
        s = str(s)
        for z, h in zip('０１２３４５６７８９', '0123456789'):
            s = s.replace(z, h)
        try: return int(s) if s.isdigit() else np.nan
        except: return np.nan
    
    df['finish_position'] = df['着順'].apply(zen_to_han)
    df = df[df['finish_position'].notna() & (df['finish_position'] > 0)]
    
    # カラム名変換
    df = df.rename(columns={
        '単勝オッズ': 'odds_win', '人気': 'popularity', '年齢': 'horse_age',
        '性別': 'horse_sex', '斤量': 'load_weight', '枠番': 'gate_no',
        '頭数': 'field_size', '芝・ダ': 'track_type', '距離': 'distance',
        '馬場状態': 'track_condition', '天気': 'weather', '場所': 'place',
        '脚質': 'running_style'
    })
    
    # 数値変換
    for col in ['odds_win', 'popularity', 'horse_age', 'load_weight', 'field_size', 'distance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 勝利フラグ
    df['is_win'] = (df['finish_position'] == 1).astype(int)
    df['is_top3'] = (df['finish_position'] <= 3).astype(int)
    
    print(f"  → {len(df):,}件読み込み完了")
    return df

def extract_patterns(df: pd.DataFrame) -> list:
    """パターンを抽出"""
    print("\n📊 パターン抽出中...")
    
    patterns = []
    baseline_win_rate = df['is_win'].mean()
    baseline_top3_rate = df['is_top3'].mean()
    
    print(f"  ベースライン勝率: {baseline_win_rate*100:.2f}%")
    print(f"  ベースラインTop3率: {baseline_top3_rate*100:.2f}%")
    
    # 1. 数値特徴量のビン別パターン
    for feature, config in FEATURE_BINS.items():
        if feature not in df.columns:
            continue
        
        bins = config['bins']
        labels = config['labels']
        
        # infを大きな数値に置換
        bins_safe = [b if b != float('inf') else 99999 for b in bins]
        
        df[f'{feature}_bin'] = pd.cut(df[feature], bins=bins_safe, labels=labels, include_lowest=True)
        
        for label in labels:
            subset = df[df[f'{feature}_bin'] == label]
            if len(subset) < 100:
                continue
            
            win_rate = subset['is_win'].mean()
            top3_rate = subset['is_top3'].mean()
            
            # 効果量計算（ポイント差）
            win_effect = (win_rate - baseline_win_rate) * 100
            top3_effect = (top3_rate - baseline_top3_rate) * 100
            
            # 統計的検定
            try:
                _, p_value = stats.ttest_ind(
                    subset['is_win'].values, 
                    df['is_win'].values
                )
            except:
                p_value = 1.0
            
            if abs(win_effect) >= 1.0 and p_value < 0.05:
                patterns.append({
                    'pattern_id': f'{feature}_{label}',
                    'conditions': {
                        feature: {
                            'min': bins[labels.index(label)],
                            'max': bins[labels.index(label) + 1],
                            'label': label
                        }
                    },
                    'sample_size': len(subset),
                    'win_rate': win_rate,
                    'top3_rate': top3_rate,
                    'win_effect': win_effect,
                    'top3_effect': top3_effect,
                    'p_value': p_value,
                    'type': 'single_feature'
                })
    
    # 2. カテゴリカル特徴量のパターン
    for feature in ['track_type', 'track_condition', 'weather']:
        if feature not in df.columns:
            continue
        
        for value in df[feature].dropna().unique():
            subset = df[df[feature] == value]
            if len(subset) < 100:
                continue
            
            win_rate = subset['is_win'].mean()
            win_effect = (win_rate - baseline_win_rate) * 100
            
            try:
                _, p_value = stats.ttest_ind(subset['is_win'].values, df['is_win'].values)
            except:
                p_value = 1.0
            
            if abs(win_effect) >= 1.0 and p_value < 0.05:
                patterns.append({
                    'pattern_id': f'{feature}_{value}',
                    'conditions': {feature: value},
                    'sample_size': len(subset),
                    'win_rate': win_rate,
                    'win_effect': win_effect,
                    'p_value': p_value,
                    'type': 'categorical'
                })
    
    # 3. 複合パターン（オッズ×人気など）
    compound_features = [
        ('odds_win', 'popularity'),
        ('odds_win', 'field_size'),
        ('track_type', 'track_condition'),
    ]
    
    for f1, f2 in compound_features:
        if f1 not in df.columns or f2 not in df.columns:
            continue
        
        # ビン化
        if f1 in FEATURE_BINS:
            bins1 = [b if b != float('inf') else 99999 for b in FEATURE_BINS[f1]['bins']]
            labels1 = FEATURE_BINS[f1]['labels']
            df[f'{f1}_bin'] = pd.cut(df[f1], bins=bins1, labels=labels1, include_lowest=True)
            col1 = f'{f1}_bin'
        else:
            col1 = f1
        
        if f2 in FEATURE_BINS:
            bins2 = [b if b != float('inf') else 99999 for b in FEATURE_BINS[f2]['bins']]
            labels2 = FEATURE_BINS[f2]['labels']
            df[f'{f2}_bin'] = pd.cut(df[f2], bins=bins2, labels=labels2, include_lowest=True)
            col2 = f'{f2}_bin'
        else:
            col2 = f2
        
        for v1 in df[col1].dropna().unique():
            for v2 in df[col2].dropna().unique():
                subset = df[(df[col1] == v1) & (df[col2] == v2)]
                if len(subset) < 500:
                    continue
                
                win_rate = subset['is_win'].mean()
                win_effect = (win_rate - baseline_win_rate) * 100
                
                try:
                    _, p_value = stats.ttest_ind(subset['is_win'].values, df['is_win'].values)
                except:
                    p_value = 1.0
                
                if abs(win_effect) >= 2.0 and p_value < 0.05:
                    patterns.append({
                        'pattern_id': f'{f1}_{v1}_{f2}_{v2}',
                        'conditions': {f1: str(v1), f2: str(v2)},
                        'sample_size': len(subset),
                        'win_rate': win_rate,
                        'win_effect': win_effect,
                        'p_value': p_value,
                        'type': 'compound'
                    })
    
    print(f"  → {len(patterns)}個のパターン候補を抽出")
    return patterns

def validate_patterns(patterns: list, df: pd.DataFrame) -> list:
    """パターンを検証（時系列で一貫性確認）"""
    print("\n🔍 パターン検証中...")
    
    validated = []
    
    # 年別に分割
    df['year'] = df['date'].str[:4]
    years = sorted(df['year'].unique())
    
    for pattern in patterns:
        # 各年での効果を計算
        yearly_effects = []
        
        for year in years[-5:]:  # 直近5年
            year_df = df[df['year'] == year]
            
            # 条件でフィルタ
            subset = year_df.copy()
            for feat, cond in pattern['conditions'].items():
                if isinstance(cond, dict):
                    # 数値範囲
                    min_val = cond['min']
                    max_val = cond['max'] if cond['max'] != float('inf') else 99999
                    subset = subset[(subset[feat] > min_val) & (subset[feat] <= max_val)]
                else:
                    # カテゴリカル
                    if f'{feat}_bin' in subset.columns:
                        subset = subset[subset[f'{feat}_bin'] == cond]
                    else:
                        subset = subset[subset[feat] == cond]
            
            if len(subset) >= 50:
                effect = (subset['is_win'].mean() - year_df['is_win'].mean()) * 100
                yearly_effects.append(effect)
        
        if len(yearly_effects) < 3:
            continue
        
        # 一貫性チェック（同じ方向の効果が続いているか）
        if pattern['win_effect'] > 0:
            consistency = sum(1 for e in yearly_effects if e > 0) / len(yearly_effects)
        else:
            consistency = sum(1 for e in yearly_effects if e < 0) / len(yearly_effects)
        
        if consistency >= 0.6:
            pattern['consistency'] = consistency
            pattern['yearly_effects'] = yearly_effects
            pattern['score_adjustment'] = min(max(pattern['win_effect'], -5), 5)  # -5 ~ +5
            validated.append(pattern)
    
    print(f"  → {len(validated)}個のパターンが検証をパス")
    return validated

def save_patterns(patterns: list):
    """パターンを保存"""
    # infをNoneに変換（JSON対応）
    for p in patterns:
        for feat, cond in p['conditions'].items():
            if isinstance(cond, dict):
                if cond.get('max') == float('inf'):
                    cond['max'] = None
    
    output_path = PATTERN_PATH / 'validated_patterns.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 保存完了: {output_path}")

def display_patterns(patterns: list):
    """パターンを表示"""
    print("\n" + "="*70)
    print("📋 検証済みパターン一覧")
    print("="*70)
    
    # 効果量でソート
    sorted_patterns = sorted(patterns, key=lambda x: abs(x['win_effect']), reverse=True)
    
    print(f"\n{'パターンID':<35} | {'サンプル':>8} | {'効果':>8} | {'一貫性':>6} | {'調整':>6}")
    print("-"*70)
    
    for p in sorted_patterns[:20]:
        effect_sign = '+' if p['win_effect'] > 0 else ''
        adj_sign = '+' if p['score_adjustment'] > 0 else ''
        print(f"{p['pattern_id'][:35]:<35} | {p['sample_size']:>8,} | {effect_sign}{p['win_effect']:>6.1f}pt | {p['consistency']:>5.0%} | {adj_sign}{p['score_adjustment']:>5.1f}")
    
    # 予測根拠として使えるパターン
    print("\n" + "="*70)
    print("💡 予測根拠として使用するパターン")
    print("="*70)
    
    positive = [p for p in sorted_patterns if p['win_effect'] > 2]
    negative = [p for p in sorted_patterns if p['win_effect'] < -2]
    
    print(f"\n✅ 勝率UP（スコア加算）: {len(positive)}個")
    for p in positive[:5]:
        print(f"  ・{p['pattern_id']}: +{p['win_effect']:.1f}pt → スコア+{p['score_adjustment']:.1f}")
    
    print(f"\n❌ 勝率DOWN（スコア減算）: {len(negative)}個")
    for p in negative[:5]:
        print(f"  ・{p['pattern_id']}: {p['win_effect']:.1f}pt → スコア{p['score_adjustment']:.1f}")

def main():
    print("="*70)
    print("🔄 自己学習サイクル実行")
    print("="*70)
    
    # データ読み込み
    df = load_historical_data()
    
    # パターン抽出
    patterns = extract_patterns(df)
    
    # パターン検証
    validated = validate_patterns(patterns, df)
    
    # 表示
    display_patterns(validated)
    
    # 保存
    save_patterns(validated)
    
    print("\n✅ 自己学習サイクル完了")

if __name__ == "__main__":
    main()
