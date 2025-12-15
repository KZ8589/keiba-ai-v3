"""
自動学習サイクル - 結果収集→分析→パターン更新を自動実行
毎週末のレース終了後に実行する想定
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from scipy import stats

from learning.config import FEATURE_BINS, DB_PATH

CSV_PATH = Path('data/csv_imports/results/20150105_20251130all.csv')
PATTERN_PATH = Path('data/patterns')
PATTERN_PATH.mkdir(parents=True, exist_ok=True)

class AutoLearningCycle:
    """自動学習サイクル"""
    
    def __init__(self):
        self.patterns_file = PATTERN_PATH / 'validated_patterns.json'
        self.history_file = PATTERN_PATH / 'learning_history.json'
    
    def load_existing_patterns(self) -> list:
        """既存パターンを読み込み"""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_patterns(self, patterns: list):
        """パターンを保存"""
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
    
    def log_learning_history(self, action: str, details: dict):
        """学習履歴を記録"""
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })
        
        # 直近100件のみ保持
        history = history[-100:]
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def load_historical_data(self, min_date: str = None) -> pd.DataFrame:
        """過去データを読み込み"""
        df = pd.read_csv(CSV_PATH, encoding='cp932', low_memory=False)
        
        df['date'] = df['日付'].apply(lambda x: f"20{str(x)[:2]}-{str(x)[2:4]}-{str(x)[4:6]}" if pd.notna(x) else None)
        
        if min_date:
            df = df[df['date'] >= min_date]
        
        def zen_to_han(s):
            if pd.isna(s): return np.nan
            s = str(s)
            for z, h in zip('０１２３４５６７８９', '0123456789'):
                s = s.replace(z, h)
            try: return int(s) if s.isdigit() else np.nan
            except: return np.nan
        
        df['finish_position'] = df['着順'].apply(zen_to_han)
        df = df[df['finish_position'].notna() & (df['finish_position'] > 0)]
        
        df = df.rename(columns={
            '単勝オッズ': 'odds_win', '人気': 'popularity', '年齢': 'horse_age',
            '頭数': 'field_size', '距離': 'distance'
        })
        
        for col in ['odds_win', 'popularity', 'horse_age', 'field_size', 'distance']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['is_win'] = (df['finish_position'] == 1).astype(int)
        df['year'] = df['date'].str[:4]
        
        return df
    
    def extract_and_validate_patterns(self, df: pd.DataFrame) -> list:
        """パターン抽出と検証を一括実行"""
        patterns = []
        baseline_win_rate = df['is_win'].mean()
        years = sorted(df['year'].unique())
        
        # 数値特徴量のビン別パターン
        for feature, config in FEATURE_BINS.items():
            if feature not in df.columns:
                continue
            
            bins = config['bins']
            labels = config['labels']
            bins_safe = [b if b != float('inf') else 99999 for b in bins]
            
            df[f'{feature}_bin'] = pd.cut(df[feature], bins=bins_safe, labels=labels, include_lowest=True)
            
            for label in labels:
                subset = df[df[f'{feature}_bin'] == label]
                if len(subset) < 100:
                    continue
                
                win_rate = subset['is_win'].mean()
                win_effect = (win_rate - baseline_win_rate) * 100
                
                try:
                    _, p_value = stats.ttest_ind(subset['is_win'].values, df['is_win'].values)
                except:
                    p_value = 1.0
                
                if abs(win_effect) < 1.0 or p_value >= 0.05:
                    continue
                
                # 年別一貫性チェック
                yearly_effects = []
                for year in years[-5:]:
                    year_df = df[df['year'] == year]
                    year_subset = year_df[year_df[f'{feature}_bin'] == label]
                    if len(year_subset) >= 50:
                        effect = (year_subset['is_win'].mean() - year_df['is_win'].mean()) * 100
                        yearly_effects.append(effect)
                
                if len(yearly_effects) < 3:
                    continue
                
                if win_effect > 0:
                    consistency = sum(1 for e in yearly_effects if e > 0) / len(yearly_effects)
                else:
                    consistency = sum(1 for e in yearly_effects if e < 0) / len(yearly_effects)
                
                if consistency >= 0.6:
                    patterns.append({
                        'pattern_id': f'{feature}_{label}',
                        'conditions': {
                            feature: {
                                'min': bins[labels.index(label)],
                                'max': bins[labels.index(label) + 1] if bins[labels.index(label) + 1] != float('inf') else None,
                                'label': label
                            }
                        },
                        'sample_size': len(subset),
                        'win_rate': win_rate,
                        'win_effect': win_effect,
                        'p_value': p_value,
                        'consistency': consistency,
                        'score_adjustment': min(max(win_effect, -5), 5),
                        'last_updated': datetime.now().isoformat(),
                        'type': 'single_feature'
                    })
        
        # 複合パターン
        compound_features = [('odds_win', 'popularity'), ('odds_win', 'field_size')]
        
        for f1, f2 in compound_features:
            if f1 not in df.columns or f2 not in df.columns:
                continue
            
            for label1 in FEATURE_BINS[f1]['labels']:
                for label2 in FEATURE_BINS[f2]['labels']:
                    subset = df[(df[f'{f1}_bin'] == label1) & (df[f'{f2}_bin'] == label2)]
                    if len(subset) < 500:
                        continue
                    
                    win_rate = subset['is_win'].mean()
                    win_effect = (win_rate - baseline_win_rate) * 100
                    
                    if abs(win_effect) < 2.0:
                        continue
                    
                    try:
                        _, p_value = stats.ttest_ind(subset['is_win'].values, df['is_win'].values)
                    except:
                        p_value = 1.0
                    
                    if p_value >= 0.05:
                        continue
                    
                    # 年別一貫性
                    yearly_effects = []
                    for year in years[-5:]:
                        year_df = df[df['year'] == year]
                        year_subset = year_df[(year_df[f'{f1}_bin'] == label1) & (year_df[f'{f2}_bin'] == label2)]
                        if len(year_subset) >= 50:
                            effect = (year_subset['is_win'].mean() - year_df['is_win'].mean()) * 100
                            yearly_effects.append(effect)
                    
                    if len(yearly_effects) < 3:
                        continue
                    
                    if win_effect > 0:
                        consistency = sum(1 for e in yearly_effects if e > 0) / len(yearly_effects)
                    else:
                        consistency = sum(1 for e in yearly_effects if e < 0) / len(yearly_effects)
                    
                    if consistency >= 0.6:
                        bins1 = FEATURE_BINS[f1]['bins']
                        labels1 = FEATURE_BINS[f1]['labels']
                        bins2 = FEATURE_BINS[f2]['bins']
                        labels2 = FEATURE_BINS[f2]['labels']
                        
                        patterns.append({
                            'pattern_id': f'{f1}_{label1}_{f2}_{label2}',
                            'conditions': {
                                f1: {
                                    'min': bins1[labels1.index(label1)],
                                    'max': bins1[labels1.index(label1) + 1] if bins1[labels1.index(label1) + 1] != float('inf') else None,
                                    'label': label1
                                },
                                f2: {
                                    'min': bins2[labels2.index(label2)],
                                    'max': bins2[labels2.index(label2) + 1] if bins2[labels2.index(label2) + 1] != float('inf') else None,
                                    'label': label2
                                }
                            },
                            'sample_size': len(subset),
                            'win_rate': win_rate,
                            'win_effect': win_effect,
                            'p_value': p_value,
                            'consistency': consistency,
                            'score_adjustment': min(max(win_effect, -5), 5),
                            'last_updated': datetime.now().isoformat(),
                            'type': 'compound'
                        })
        
        return patterns
    
    def run_full_cycle(self, incremental: bool = True):
        """
        完全な学習サイクルを実行
        
        Args:
            incremental: True=直近1年のみ、False=全期間
        """
        print("="*70)
        print(f"🔄 自動学習サイクル実行 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        print("="*70)
        
        # データ読み込み
        min_date = None
        if incremental:
            min_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        print(f"\n📚 データ読み込み中... (min_date: {min_date or '全期間'})")
        df = self.load_historical_data(min_date)
        print(f"  → {len(df):,}件")
        
        # パターン抽出・検証
        print("\n📊 パターン抽出・検証中...")
        new_patterns = self.extract_and_validate_patterns(df)
        print(f"  → {len(new_patterns)}個のパターンを検出")
        
        # 既存パターンとマージ
        existing_patterns = self.load_existing_patterns()
        existing_ids = {p['pattern_id'] for p in existing_patterns}
        new_ids = {p['pattern_id'] for p in new_patterns}
        
        # 更新されたパターン
        updated_count = len(existing_ids & new_ids)
        added_count = len(new_ids - existing_ids)
        removed_count = len(existing_ids - new_ids)
        
        # 新パターンで上書き
        self.save_patterns(new_patterns)
        
        # 履歴記録
        self.log_learning_history('full_cycle', {
            'data_count': len(df),
            'pattern_count': len(new_patterns),
            'updated': updated_count,
            'added': added_count,
            'removed': removed_count
        })
        
        # 結果表示
        print(f"\n📈 パターン更新結果:")
        print(f"  更新: {updated_count}個")
        print(f"  新規: {added_count}個")
        print(f"  削除: {removed_count}個")
        print(f"  合計: {len(new_patterns)}個")
        
        # Top5パターン表示
        print(f"\n🏆 効果量Top5パターン:")
        sorted_patterns = sorted(new_patterns, key=lambda x: abs(x['win_effect']), reverse=True)
        for p in sorted_patterns[:5]:
            sign = '+' if p['win_effect'] > 0 else ''
            print(f"  {p['pattern_id']}: {sign}{p['win_effect']:.1f}pt (一貫性{p['consistency']:.0%})")
        
        print(f"\n✅ 学習サイクル完了")
        print(f"💾 保存先: {self.patterns_file}")
        
        return {
            'pattern_count': len(new_patterns),
            'updated': updated_count,
            'added': added_count,
            'removed': removed_count
        }

def main():
    cycle = AutoLearningCycle()
    cycle.run_full_cycle(incremental=False)  # 全期間で実行

if __name__ == "__main__":
    main()
