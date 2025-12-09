"""
パターン検証モジュール
pattern_candidates をバックテストで検証し、validated_patterns に登録
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database import get_connection
from src.core.config import LEARNING_CONFIG


class PatternValidator:
    """パターン候補をバックテストで検証するクラス"""
    
    def __init__(
        self,
        min_sample_size: int = None,
        min_p_value: float = None,
        min_effect_size: float = None,
        min_validation_periods: int = None
    ):
        """
        Args:
            min_sample_size: 最小サンプルサイズ（デフォルト: LEARNING_CONFIG）
            min_p_value: p値の閾値（デフォルト: LEARNING_CONFIG）
            min_effect_size: 最小効果量（デフォルト: LEARNING_CONFIG）
            min_validation_periods: 最小検証期間数（デフォルト: LEARNING_CONFIG）
        """
        self.min_sample_size = min_sample_size or LEARNING_CONFIG['min_sample_size']
        self.min_p_value = min_p_value or LEARNING_CONFIG['min_p_value']
        self.min_effect_size = min_effect_size or LEARNING_CONFIG['min_effect_size']
        self.min_validation_periods = min_validation_periods or LEARNING_CONFIG['min_validation_periods']
        
        self.validated_count = 0
        self.rejected_count = 0
    
    def get_pending_candidates(self) -> list:
        """
        検証待ちのパターン候補を取得
        
        Returns:
            list: パターン候補のリスト
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, pattern_type, pattern_name, pattern_conditions,
                       extraction_method, sample_size, effect_size, p_value
                FROM pattern_candidates
                WHERE validation_status = 'pending'
                ORDER BY ABS(effect_size) DESC
            """)
            
            candidates = []
            for row in cursor.fetchall():
                candidates.append({
                    'id': row[0],
                    'pattern_type': row[1],
                    'pattern_name': row[2],
                    'pattern_conditions': json.loads(row[3]) if row[3] else {},
                    'extraction_method': row[4],
                    'sample_size': row[5],
                    'effect_size': row[6],
                    'p_value': row[7]
                })
            
            return candidates
    
    def get_historical_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        バックテスト用の過去データを取得
        
        Args:
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            DataFrame: 過去データ
        """
        with get_connection() as conn:
            query = """
                SELECT 
                    rr.race_id,
                    rr.date,
                    rr.place_code,
                    rr.horse_no,
                    rr.horse_name,
                    rr.finish_position,
                    rr.odds_win,
                    rr.popularity,
                    rr.horse_age,
                    rr.field_size,
                    rd.track_type,
                    rd.distance,
                    rd.track_condition,
                    rd.weather
                FROM race_results rr
                LEFT JOIN race_details rd ON rr.race_id = rd.race_id
                WHERE rr.finish_position IS NOT NULL
                  AND rr.finish_position > 0
                  AND rr.odds_win IS NOT NULL
            """
            
            conditions = []
            params = []
            
            if start_date:
                conditions.append("rr.date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("rr.date <= ?")
                params.append(end_date)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            df = pd.read_sql_query(query, conn, params=params if params else None)
        
        return df
    
    def apply_pattern_conditions(self, df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
        """
        パターン条件を適用してフィルタリング
        
        Args:
            df: データフレーム
            conditions: パターン条件
        
        Returns:
            DataFrame: フィルタリング後のデータ
        """
        filtered = df.copy()
        
        for key, value in conditions.items():
            if key not in filtered.columns:
                continue
            
            if isinstance(value, list):
                # リストの場合はOR条件
                filtered = filtered[filtered[key].isin(value)]
            elif isinstance(value, str):
                # 文字列の場合は完全一致
                if value.startswith('>='):
                    filtered = filtered[filtered[key] >= float(value[2:])]
                elif value.startswith('<='):
                    filtered = filtered[filtered[key] <= float(value[2:])]
                elif value.startswith('>'):
                    filtered = filtered[filtered[key] > float(value[1:])]
                elif value.startswith('<'):
                    filtered = filtered[filtered[key] < float(value[1:])]
                else:
                    filtered = filtered[filtered[key].astype(str) == value]
            elif isinstance(value, (int, float)):
                filtered = filtered[filtered[key] == value]
        
        return filtered
    
    def validate_pattern(self, candidate: dict, df: pd.DataFrame) -> dict:
        """
        単一パターンを検証
        
        Args:
            candidate: パターン候補
            df: バックテスト用データ
        
        Returns:
            dict: 検証結果
        """
        conditions = candidate['pattern_conditions']
        
        # パターン条件に該当するデータを抽出
        pattern_df = self.apply_pattern_conditions(df, conditions)
        other_df = df[~df.index.isin(pattern_df.index)]
        
        # サンプルサイズチェック
        if len(pattern_df) < self.min_sample_size:
            return {
                'status': 'rejected',
                'reason': f'サンプルサイズ不足（{len(pattern_df)} < {self.min_sample_size}）',
                'sample_size': len(pattern_df)
            }
        
        # 勝率計算
        pattern_win_rate = (pattern_df['finish_position'] == 1).mean() * 100
        baseline_win_rate = (df['finish_position'] == 1).mean() * 100
        effect_size = pattern_win_rate - baseline_win_rate
        
        # 統計検定（カイ二乗検定）
        try:
            table = [
                [(pattern_df['finish_position'] == 1).sum(), 
                 (pattern_df['finish_position'] != 1).sum()],
                [(other_df['finish_position'] == 1).sum(), 
                 (other_df['finish_position'] != 1).sum()]
            ]
            chi2, p_value, dof, expected = stats.chi2_contingency(table)
        except ValueError:
            p_value = 1.0
        
        # 複数期間での検証
        period_results = self._validate_by_periods(pattern_df, df)
        
        # 判定
        is_valid = (
            abs(effect_size) >= self.min_effect_size and
            p_value < self.min_p_value and
            period_results['consistent_periods'] >= self.min_validation_periods
        )
        
        return {
            'status': 'validated' if is_valid else 'rejected',
            'reason': None if is_valid else self._get_rejection_reason(
                effect_size, p_value, period_results['consistent_periods']
            ),
            'sample_size': len(pattern_df),
            'win_rate': round(pattern_win_rate, 2),
            'baseline_rate': round(baseline_win_rate, 2),
            'effect_size': round(effect_size, 2),
            'p_value': round(p_value, 4),
            'validation_periods': period_results['total_periods'],
            'consistent_periods': period_results['consistent_periods'],
            'validation_consistency': period_results['consistency'],
            'period_details': period_results['details']
        }
    
    def _validate_by_periods(self, pattern_df: pd.DataFrame, full_df: pd.DataFrame) -> dict:
        """
        複数期間で効果の一貫性を検証
        
        Args:
            pattern_df: パターン適用データ
            full_df: 全データ
        
        Returns:
            dict: 期間別検証結果
        """
        if 'date' not in pattern_df.columns or pattern_df.empty:
            return {
                'total_periods': 0,
                'consistent_periods': 0,
                'consistency': 0.0,
                'details': []
            }
        
        # 日付をdatetimeに変換
        pattern_df = pattern_df.copy()
        pattern_df['date'] = pd.to_datetime(pattern_df['date'])
        
        full_df = full_df.copy()
        full_df['date'] = pd.to_datetime(full_df['date'])
        
        # 四半期ごとに分割
        pattern_df['quarter'] = pattern_df['date'].dt.to_period('Q')
        full_df['quarter'] = full_df['date'].dt.to_period('Q')
        
        quarters = sorted(pattern_df['quarter'].unique())
        
        details = []
        consistent_count = 0
        baseline_win_rate = (full_df['finish_position'] == 1).mean() * 100
        
        for q in quarters:
            q_pattern = pattern_df[pattern_df['quarter'] == q]
            q_full = full_df[full_df['quarter'] == q]
            
            if len(q_pattern) < 10:  # 最小サンプル
                continue
            
            q_win_rate = (q_pattern['finish_position'] == 1).mean() * 100
            q_baseline = (q_full['finish_position'] == 1).mean() * 100
            q_effect = q_win_rate - q_baseline
            
            # 効果の方向が一貫しているか
            is_consistent = (q_effect > 0) == (baseline_win_rate < q_win_rate)
            
            if is_consistent and abs(q_effect) >= self.min_effect_size * 0.5:
                consistent_count += 1
            
            details.append({
                'period': str(q),
                'sample_size': int(len(q_pattern)),
                'win_rate': round(float(q_win_rate), 2),
                'effect': round(float(q_effect), 2),
                'consistent': bool(is_consistent)
            })
        
        total_periods = len(details)
        consistency = consistent_count / total_periods if total_periods > 0 else 0
        
        return {
            'total_periods': total_periods,
            'consistent_periods': consistent_count,
            'consistency': round(consistency, 2),
            'details': details
        }
    
    def _get_rejection_reason(self, effect_size: float, p_value: float, consistent_periods: int) -> str:
        """棄却理由を生成"""
        reasons = []
        
        if abs(effect_size) < self.min_effect_size:
            reasons.append(f'効果量不足（{effect_size:.1f} < {self.min_effect_size}）')
        
        if p_value >= self.min_p_value:
            reasons.append(f'p値超過（{p_value:.3f} >= {self.min_p_value}）')
        
        if consistent_periods < self.min_validation_periods:
            reasons.append(f'一貫性不足（{consistent_periods} < {self.min_validation_periods}期間）')
        
        return '、'.join(reasons) if reasons else '検証基準未達'
    
    def register_validated_pattern(self, candidate: dict, validation_result: dict) -> int:
        """
        検証済みパターンを登録
        
        Args:
            candidate: パターン候補
            validation_result: 検証結果
        
        Returns:
            int: 登録されたパターンID
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # アクション値の計算
            action_value = validation_result['effect_size'] / 10  # 効果量の10%をスコア調整
            action_type = 'score_adjust' if candidate['pattern_type'] == 'horse' else 'confidence'
            
            cursor.execute("""
                INSERT OR REPLACE INTO validated_patterns (
                    pattern_type, pattern_name, pattern_description,
                    pattern_conditions, action_type, action_value, action_description,
                    sample_size, win_rate, baseline_rate, effect_size, p_value,
                    validation_method, validation_periods, validation_consistency,
                    reasoning, evidence, is_active, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                candidate['pattern_type'],
                candidate['pattern_name'],
                f"{candidate['pattern_name']}パターン",
                json.dumps(candidate['pattern_conditions'], ensure_ascii=False),
                action_type,
                round(action_value, 2),
                f"{'スコア' if action_type == 'score_adjust' else '信頼度'}{'+' if action_value > 0 else ''}{action_value:.1f}調整",
                validation_result['sample_size'],
                validation_result['win_rate'],
                validation_result['baseline_rate'],
                validation_result['effect_size'],
                validation_result['p_value'],
                'backtest',
                validation_result['validation_periods'],
                validation_result['validation_consistency'],
                f"バックテストで{validation_result['consistent_periods']}期間一貫した効果を確認",
                json.dumps(validation_result['period_details'], ensure_ascii=False),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            pattern_id = cursor.lastrowid
            conn.commit()
            
            return pattern_id
    
    def update_candidate_status(self, candidate_id: int, status: str, result: dict):
        """
        候補のステータスを更新
        
        Args:
            candidate_id: 候補ID
            status: 新ステータス
            result: 検証結果
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE pattern_candidates
                SET validation_status = ?,
                    validation_result = ?
                WHERE id = ?
            """, (
                status,
                json.dumps(result, ensure_ascii=False),
                candidate_id
            ))
            conn.commit()
    
    def validate_all(
        self,
        start_date: str = '2022-01-01',
        end_date: str = None
    ) -> dict:
        """
        全候補を検証
        
        Args:
            start_date: バックテスト開始日
            end_date: バックテスト終了日
        
        Returns:
            dict: 検証結果サマリー
        """
        self.validated_count = 0
        self.rejected_count = 0
        
        print("="*60)
        print("🔍 パターン検証開始")
        print("="*60)
        
        # 候補取得
        candidates = self.get_pending_candidates()
        
        if not candidates:
            print("  検証待ち候補なし")
            return {
                'validated': 0,
                'rejected': 0,
                'total': 0
            }
        
        print(f"  検証対象: {len(candidates)}件")
        print(f"  検証期間: {start_date} 〜 {end_date or '最新'}")
        print()
        
        # バックテスト用データ取得
        print("📊 バックテスト用データ読み込み中...")
        df = self.get_historical_data(start_date, end_date)
        print(f"  → {len(df):,}レコード")
        print()
        
        if df.empty:
            print("❌ バックテスト用データなし")
            return {
                'validated': 0,
                'rejected': 0,
                'total': len(candidates)
            }
        
        # 各候補を検証
        validated_patterns = []
        rejected_patterns = []
        
        for candidate in candidates:
            print(f"検証中: {candidate['pattern_name']}...")
            
            result = self.validate_pattern(candidate, df)
            
            if result['status'] == 'validated':
                # 検証済みパターンを登録
                pattern_id = self.register_validated_pattern(candidate, result)
                self.update_candidate_status(candidate['id'], 'validated', result)
                validated_patterns.append({
                    'name': candidate['pattern_name'],
                    'effect': result['effect_size'],
                    'p_value': result['p_value'],
                    'pattern_id': pattern_id
                })
                self.validated_count += 1
                print(f"  ✅ 検証合格（効果: {result['effect_size']:+.1f}pt, p={result['p_value']:.3f}）")
            else:
                self.update_candidate_status(candidate['id'], 'rejected', result)
                rejected_patterns.append({
                    'name': candidate['pattern_name'],
                    'reason': result['reason']
                })
                self.rejected_count += 1
                print(f"  ❌ 棄却（{result['reason']}）")
        
        # 結果サマリー
        print()
        print("-"*60)
        print(f"✅ 検証完了")
        print(f"  合格: {self.validated_count}件")
        print(f"  棄却: {self.rejected_count}件")
        
        if validated_patterns:
            print()
            print("📋 検証合格パターン:")
            for p in validated_patterns:
                print(f"  - {p['name']} (効果: {p['effect']:+.1f}pt)")
        
        print("="*60)
        
        return {
            'validated': self.validated_count,
            'rejected': self.rejected_count,
            'total': len(candidates),
            'validated_patterns': validated_patterns,
            'rejected_patterns': rejected_patterns
        }
    
    def get_active_patterns(self) -> list:
        """
        アクティブな検証済みパターンを取得
        
        Returns:
            list: パターンのリスト
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, pattern_type, pattern_name, pattern_conditions,
                       action_type, action_value, win_rate, effect_size
                FROM validated_patterns
                WHERE is_active = 1
                ORDER BY ABS(effect_size) DESC
            """)
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'id': row[0],
                    'pattern_type': row[1],
                    'pattern_name': row[2],
                    'pattern_conditions': json.loads(row[3]) if row[3] else {},
                    'action_type': row[4],
                    'action_value': row[5],
                    'win_rate': row[6],
                    'effect_size': row[7]
                })
            
            return patterns


def test_validator():
    """テスト関数"""
    print("="*60)
    print("🧪 PatternValidator テスト")
    print("="*60)
    
    # 前回のテストデータをクリーンアップ
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pattern_candidates WHERE extraction_method = 'test'")
        cursor.execute("DELETE FROM validated_patterns WHERE pattern_name LIKE 'test_%'")
        conn.commit()
    print("  前回テストデータ削除")
    
    # テスト用の候補を手動で追加
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # テスト用候補を追加
        test_candidates = [
            {
                'pattern_type': 'condition',
                'pattern_name': 'test_track_condition_良',
                'pattern_conditions': {'track_condition': '良'},
                'extraction_method': 'test'
            },
            {
                'pattern_type': 'horse',
                'pattern_name': 'test_popularity_低',
                'pattern_conditions': {'popularity': '1'},
                'extraction_method': 'test'
            }
        ]
        
        for c in test_candidates:
            cursor.execute("""
                INSERT INTO pattern_candidates (
                    pattern_type, pattern_name, pattern_conditions,
                    extraction_method, extraction_date,
                    sample_size, effect_size, validation_status
                ) VALUES (?, ?, ?, ?, ?, 100, 5.0, 'pending')
            """, (
                c['pattern_type'],
                c['pattern_name'],
                json.dumps(c['pattern_conditions']),
                c['extraction_method'],
                datetime.now().strftime('%Y-%m-%d')
            ))
        
        conn.commit()
    
    print("  テスト用候補を追加")
    
    # バリデータ実行（テスト用に閾値を下げる）
    validator = PatternValidator(
        min_sample_size=10,
        min_p_value=0.5,
        min_effect_size=0.1,
        min_validation_periods=1
    )
    
    result = validator.validate_all(start_date='2024-01-01', end_date='2024-12-31')
    
    print()
    print(f"📊 テスト結果:")
    print(f"  検証合格: {result['validated']}")
    print(f"  棄却: {result['rejected']}")
    
    # アクティブパターン確認
    active = validator.get_active_patterns()
    print(f"  アクティブパターン数: {len(active)}")
    
    if active:
        print()
        print("📋 アクティブパターン:")
        for p in active[:3]:
            print(f"  - {p['pattern_name']} | 効果: {p['effect_size']:+.1f}pt | アクション: {p['action_value']:+.2f}")
    
    # クリーンアップ
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pattern_candidates WHERE extraction_method = 'test'")
        cursor.execute("DELETE FROM validated_patterns WHERE pattern_name LIKE 'test_%'")
        conn.commit()
    print()
    print("  テストデータ削除完了")
    print()
    print("✅ テスト完了")


if __name__ == "__main__":
    test_validator()


