"""
pattern_applier.py - 検証済みパターンを予測スコアに適用

使用方法:
    from learning.pattern_applier import PatternApplier
    applier = PatternApplier()
    df, applied = applier.apply_patterns(df)
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path
from .config import get_bin_range

DB_PATH = Path(__file__).parent.parent.parent / 'data' / 'keiba.db'


class PatternApplier:
    """検証済みパターンを予測に適用するクラス"""
    
    def __init__(self):
        self.patterns = []
        self.load_patterns()
    
    def load_patterns(self):
        """アクティブなパターンをDBから読み込み"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, pattern_type, pattern_name, pattern_conditions, 
                   effect_size, action_type, action_value
            FROM validated_patterns
            WHERE is_active = 1
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        self.patterns = []
        for row in rows:
            pattern = {
                'id': row[0],
                'type': row[1],
                'name': row[2],
                'conditions': json.loads(row[3]) if row[3] else {},
                'effect_size': row[4] or 0,
                'action_type': row[5] or 'score_adjustment',
                'action_value': row[6] or row[4] or 0  # action_valueがなければeffect_sizeを使用
            }
            self.patterns.append(pattern)
        
        print(f"📚 検証済みパターン読み込み: {len(self.patterns)}件")
    
    def check_condition(self, row: pd.Series, conditions: dict) -> bool:
        """
        レコードがパターン条件にマッチするか判定
        
        Args:
            row: DataFrameの1行
            conditions: パターン条件 {"track_type": "ダート", ...}
        
        Returns:
            bool: マッチすればTrue
        """
        for col, value in conditions.items():
            if col not in row.index:
                return False
            
            row_value = row[col]
            
            # 範囲条件の場合（将来拡張用）
            if isinstance(value, dict):
                if 'min' in value and row_value < value['min']:
                    return False
                if 'max' in value and row_value > value['max']:
                    return False
            # 単一値条件
            elif row_value != value:
                return False
        
        return True
    
    def apply_patterns(self, df: pd.DataFrame) -> tuple:
        """
        DataFrameにパターンを適用してスコアを調整
        
        Args:
            df: 予測結果DataFrame（score列必須）
        
        Returns:
            tuple: (調整済みdf, 適用パターン情報dict)
        """
        if not self.patterns:
            return df, {'applied': 0, 'patterns': []}
        
        df = df.copy()
        applied_info = {
            'applied': 0,
            'patterns': [],
            'adjustments': []
        }
        
        # 元のスコアを保存
        df['score_original'] = df['score']
        df['applied_patterns'] = ''
        
        for pattern in self.patterns:
            pattern_name = pattern['name']
            conditions = pattern['conditions']
            effect = pattern['action_value']
            
            # 各レコードに対してパターンをチェック
            matches = df.apply(lambda row: self.check_condition(row, conditions), axis=1)
            match_count = matches.sum()
            
            if match_count > 0:
                # スコア調整
                df.loc[matches, 'score'] += effect
                
                # 適用パターン記録
                df.loc[matches, 'applied_patterns'] += f"{pattern_name}({effect:+.1f}), "
                
                applied_info['patterns'].append({
                    'name': pattern_name,
                    'effect': effect,
                    'matched': int(match_count)
                })
                applied_info['applied'] += 1
                
                print(f"  ✅ {pattern_name}: {match_count}件に適用 ({effect:+.1f}pt)")
        
        # スコアを0-100にクリップ
        df['score'] = df['score'].clip(0, 100)
        
        # 適用パターン文字列の末尾カンマを削除
        df['applied_patterns'] = df['applied_patterns'].str.rstrip(', ')
        
        # スコア変動があればランクを再計算
        if applied_info['applied'] > 0:
            df['pred_rank'] = df.groupby('race_id')['score'].rank(ascending=False, method='first').astype(int)
        
        return df, applied_info
    
    def get_pattern_summary(self) -> str:
        """読み込み済みパターンのサマリーを返す"""
        if not self.patterns:
            return "アクティブなパターンなし"
        
        lines = []
        for p in self.patterns:
            lines.append(f"  - {p['name']}: {p['action_value']:+.1f}pt ({p['conditions']})")
        return "\n".join(lines)


def test_applier():
    """テスト"""
    print("="*60)
    print("🧪 PatternApplier テスト")
    print("="*60)
    
    applier = PatternApplier()
    
    print("\n📋 読み込み済みパターン:")
    print(applier.get_pattern_summary())
    
    # テストデータ
    test_df = pd.DataFrame({
        'race_id': ['R1', 'R1', 'R2', 'R2'],
        'horse_no': [1, 2, 1, 2],
        'track_type': ['芝', 'ダート', '芝', 'ダート'],
        'track_condition': ['良', '良', '不良', '不良'],
        'running_style': ['逃げ', '先行', '中団', '逃げ'],
        'prev_finish': [3, 2, 1, 2],
        'score': [70.0, 65.0, 80.0, 75.0]
    })
    
    print("\n📊 適用前:")
    print(test_df[['race_id', 'horse_no', 'track_type', 'score']])
    
    result_df, applied = applier.apply_patterns(test_df)
    
    print("\n📊 適用後:")
    print(result_df[['race_id', 'horse_no', 'track_type', 'score_original', 'score', 'applied_patterns']])
    
    print(f"\n✅ 適用パターン数: {applied['applied']}")


if __name__ == "__main__":
    test_applier()
