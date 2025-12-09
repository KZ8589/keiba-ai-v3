"""
パターン抽出モジュール
予測結果から成功/失敗パターンを自動抽出
"""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.database import get_connection

# 特徴量定義
CATEGORICAL_FEATURES = [
    'track_type',        # 芝/ダート
    'track_condition',   # 良/稍重/重/不良
    'weather',           # 晴/曇/雨
    'place_code',        # 競馬場
]

NUMERICAL_BINS = {
    'odds_win': {
        'bins': [0, 3, 10, 30, 1000],
        'labels': ['低オッズ(~3)', '中オッズ(3-10)', '高オッズ(10-30)', '超高オッズ(30+)']
    },
    'popularity': {
        'bins': [0, 3, 6, 100],
        'labels': ['上位人気(1-3)', '中位人気(4-6)', '下位人気(7+)']
    },
    'field_size': {
        'bins': [0, 10, 14, 100],
        'labels': ['少頭数(~10)', '標準(11-14)', '多頭数(15+)']
    },
    'distance': {
        'bins': [0, 1400, 1800, 2200, 10000],
        'labels': ['短距離(~1400)', 'マイル(1401-1800)', '中距離(1801-2200)', '長距離(2201+)']
    },
    'horse_age': {
        'bins': [0, 3, 5, 100],
        'labels': ['若馬(2-3)', '中堅(4-5)', '古馬(6+)']
    }
}


class PatternExtractor:
    """パターン抽出クラス"""
    
    def __init__(self, min_sample_size: int = 50, min_effect_size: float = 2.0):
        """
        Args:
            min_sample_size: 最小サンプルサイズ
            min_effect_size: 最小効果量（ポイント差）
        """
        self.min_sample_size = min_sample_size
        self.min_effect_size = min_effect_size
        self.candidates = []
    
    def get_analysis_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        分析用データを取得（prediction_logs + prediction_results 結合）
        
        Args:
            start_date: 開始日（オプション）
            end_date: 終了日（オプション）
        
        Returns:
            DataFrame: 分析用データ
        """
        with get_connection() as conn:
            query = """
                SELECT 
                    pl.race_id,
                    pl.race_date,
                    pl.place_code,
                    pl.track_type,
                    pl.distance,
                    pl.track_condition,
                    pl.weather,
                    pl.field_size,
                    pl.horse_no,
                    pl.horse_name,
                    pl.pred_score,
                    pl.pred_rank,
                    pl.odds_win,
                    pl.popularity,
                    pr.is_hit_1st,
                    pr.is_hit_top3,
                    pr.upset_level,
                    pr.actual_1st_odds,
                    pr.actual_1st_popularity
                FROM prediction_logs pl
                INNER JOIN prediction_results pr ON pl.race_id = pr.race_id
                WHERE pl.pred_rank = 1
            """
            
            conditions = []
            params = []
            
            if start_date:
                conditions.append("pl.race_date >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("pl.race_date <= ?")
                params.append(end_date)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            df = pd.read_sql_query(query, conn, params=params if params else None)
        
        return df
    
    def _bin_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値特徴量をビン化"""
        df = df.copy()
        
        for col, config in NUMERICAL_BINS.items():
            if col in df.columns:
                df[f'{col}_bin'] = pd.cut(
                    df[col],
                    bins=config['bins'],
                    labels=config['labels'],
                    include_lowest=True
                )
        
        return df
    
    def extract_by_frequency(self, df: pd.DataFrame) -> list:
        """
        頻度分析による単一条件パターン抽出
        
        Args:
            df: 分析用DataFrame
        
        Returns:
            list: パターン候補リスト
        """
        candidates = []
        df = self._bin_numerical_features(df)
        
        # カテゴリ特徴量 + ビン化した数値特徴量
        features_to_analyze = CATEGORICAL_FEATURES.copy()
        features_to_analyze += [f'{col}_bin' for col in NUMERICAL_BINS.keys()]
        
        # 全体の的中率（ベースライン）
        baseline_hit_rate = df['is_hit_1st'].mean() * 100 if len(df) > 0 else 0
        baseline_top3_rate = df['is_hit_top3'].mean() * 100 if len(df) > 0 else 0
        
        for feature in features_to_analyze:
            if feature not in df.columns:
                continue
            
            # グループごとの集計
            grouped = df.groupby(feature, observed=False).agg({
                'is_hit_1st': ['sum', 'count', 'mean'],
                'is_hit_top3': ['mean']
            }).reset_index()
            
            grouped.columns = [feature, 'hit_count', 'total', 'hit_rate', 'top3_rate']
            grouped['hit_rate'] = grouped['hit_rate'] * 100
            grouped['top3_rate'] = grouped['top3_rate'] * 100
            
            for _, row in grouped.iterrows():
                if pd.isna(row[feature]) or row['total'] < self.min_sample_size:
                    continue
                
                effect_size = row['hit_rate'] - baseline_hit_rate
                
                # 効果量が閾値以上の場合のみ候補に
                if abs(effect_size) >= self.min_effect_size:
                    # パターン名生成
                    feature_name = feature.replace('_bin', '')
                    pattern_name = f"{feature_name}_{row[feature]}"
                    
                    # Type判定（馬属性 or レース条件）
                    horse_features = ['popularity', 'odds_win', 'horse_age']
                    pattern_type = 'horse' if feature_name in horse_features else 'condition'
                    
                    # アクションタイプ
                    action_type = 'score_adjust' if pattern_type == 'horse' else 'confidence'
                    
                    # アクション値（効果量に基づく）
                    action_value = round(effect_size / 10, 2)  # 10pt差 → 0.1調整
                    
                    candidates.append({
                        'pattern_type': pattern_type,
                        'pattern_name': pattern_name,
                        'pattern_conditions': {feature_name: str(row[feature])},
                        'extraction_method': 'frequency',
                        'action_type': action_type,
                        'action_value': action_value,
                        'sample_size': int(row['total']),
                        'hit_rate': round(row['hit_rate'], 2),
                        'baseline_rate': round(baseline_hit_rate, 2),
                        'effect_size': round(effect_size, 2),
                        'p_value': None,  # 頻度分析ではp値なし
                        'reasoning': f"{feature_name}={row[feature]}時の的中率{row['hit_rate']:.1f}%（基準{baseline_hit_rate:.1f}%）"
                    })
        
        return candidates
    
    def extract_by_statistics(self, df: pd.DataFrame) -> list:
        """
        統計的比較によるパターン抽出（カイ二乗検定）
        
        Args:
            df: 分析用DataFrame
        
        Returns:
            list: パターン候補リスト
        """
        candidates = []
        df = self._bin_numerical_features(df)
        
        features_to_analyze = CATEGORICAL_FEATURES.copy()
        features_to_analyze += [f'{col}_bin' for col in NUMERICAL_BINS.keys()]
        
        baseline_hit_rate = df['is_hit_1st'].mean() * 100 if len(df) > 0 else 0
        
        for feature in features_to_analyze:
            if feature not in df.columns:
                continue
            
            # 各カテゴリで検定
            for category in df[feature].dropna().unique():
                mask = df[feature] == category
                group_df = df[mask]
                other_df = df[~mask]
                
                if len(group_df) < self.min_sample_size or len(other_df) < self.min_sample_size:
                    continue
                
                # 2x2分割表
                # [グループ的中, グループ不的中]
                # [その他的中, その他不的中]
                table = [
                    [group_df['is_hit_1st'].sum(), len(group_df) - group_df['is_hit_1st'].sum()],
                    [other_df['is_hit_1st'].sum(), len(other_df) - other_df['is_hit_1st'].sum()]
                ]
                
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(table)
                except ValueError:
                    continue
                
                group_hit_rate = group_df['is_hit_1st'].mean() * 100
                effect_size = group_hit_rate - baseline_hit_rate
                
                # p < 0.1 かつ 効果量が閾値以上
                if p_value < 0.1 and abs(effect_size) >= self.min_effect_size:
                    feature_name = feature.replace('_bin', '')
                    pattern_name = f"{feature_name}_{category}_stat"
                    
                    horse_features = ['popularity', 'odds_win', 'horse_age']
                    pattern_type = 'horse' if feature_name in horse_features else 'condition'
                    action_type = 'score_adjust' if pattern_type == 'horse' else 'confidence'
                    action_value = round(effect_size / 10, 2)
                    
                    candidates.append({
                        'pattern_type': pattern_type,
                        'pattern_name': pattern_name,
                        'pattern_conditions': {feature_name: str(category)},
                        'extraction_method': 'chi_square',
                        'action_type': action_type,
                        'action_value': action_value,
                        'sample_size': len(group_df),
                        'hit_rate': round(group_hit_rate, 2),
                        'baseline_rate': round(baseline_hit_rate, 2),
                        'effect_size': round(effect_size, 2),
                        'p_value': round(p_value, 4),
                        'reasoning': f"{feature_name}={category}の的中率{group_hit_rate:.1f}%（p={p_value:.3f}）"
                    })
        
        return candidates
    
    def extract_by_decision_tree(self, df: pd.DataFrame, max_depth: int = 3) -> list:
        """
        決定木による複合条件パターン抽出
        
        Args:
            df: 分析用DataFrame
            max_depth: 決定木の最大深さ
        
        Returns:
            list: パターン候補リスト
        """
        candidates = []
        
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            print("⚠️  sklearn未インストール。決定木分析をスキップ")
            return candidates
        
        df = self._bin_numerical_features(df.copy())
        
        # 特徴量準備
        feature_cols = []
        encoders = {}
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].notna().sum() > 0:
                feature_cols.append(col)
                le = LabelEncoder()
                df[f'{col}_enc'] = le.fit_transform(df[col].fillna('unknown').astype(str))
                encoders[col] = le
        
        for col in NUMERICAL_BINS.keys():
            bin_col = f'{col}_bin'
            if bin_col in df.columns and df[bin_col].notna().sum() > 0:
                feature_cols.append(bin_col)
                le = LabelEncoder()
                df[f'{bin_col}_enc'] = le.fit_transform(df[bin_col].astype(str).fillna('unknown'))
                encoders[bin_col] = le
        
        if not feature_cols:
            return candidates
        
        # 特徴量行列
        X_cols = [f'{col}_enc' for col in feature_cols]
        X = df[X_cols].values
        y = df['is_hit_1st'].values
        
        # 決定木学習
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=self.min_sample_size,
            random_state=42
        )
        model.fit(X, y)
        
        # 葉ノードの条件を抽出
        tree = model.tree_
        baseline_hit_rate = y.mean() * 100
        
        def extract_rules(node_id, conditions):
            """再帰的にルールを抽出"""
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # 葉ノード
                samples = tree.n_node_samples[node_id]
                hit_rate = tree.value[node_id][0][1] / samples * 100 if samples > 0 else 0
                effect_size = hit_rate - baseline_hit_rate
                
                if samples >= self.min_sample_size and abs(effect_size) >= self.min_effect_size:
                    # 条件を人間が読める形式に変換
                    readable_conditions = {}
                    for cond in conditions:
                        col_idx, threshold, direction = cond
                        col_name = feature_cols[col_idx]
                        original_col = col_name.replace('_bin', '')
                        
                        if col_name in encoders:
                            le = encoders[col_name]
                            if direction == '<=':
                                values = [le.classes_[i] for i in range(int(threshold) + 1)]
                            else:
                                values = [le.classes_[i] for i in range(int(threshold) + 1, len(le.classes_))]
                            readable_conditions[original_col] = values
                    
                    if readable_conditions:
                        pattern_name = "_".join([f"{k}_{v[0] if isinstance(v, list) and len(v)==1 else 'multi'}" 
                                                  for k, v in readable_conditions.items()])[:50] + "_tree"
                        
                        candidates.append({
                            'pattern_type': 'condition',
                            'pattern_name': pattern_name,
                            'pattern_conditions': readable_conditions,
                            'extraction_method': 'decision_tree',
                            'action_type': 'confidence',
                            'action_value': round(effect_size / 10, 2),
                            'sample_size': samples,
                            'hit_rate': round(hit_rate, 2),
                            'baseline_rate': round(baseline_hit_rate, 2),
                            'effect_size': round(effect_size, 2),
                            'p_value': None,
                            'reasoning': f"決定木ルール: 的中率{hit_rate:.1f}%（基準{baseline_hit_rate:.1f}%）"
                        })
                return
            
            # 内部ノード - 子ノードへ再帰
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            # 左の子（<=）
            left_conds = conditions + [(feature_idx, threshold, '<=')]
            extract_rules(tree.children_left[node_id], left_conds)
            
            # 右の子（>）
            right_conds = conditions + [(feature_idx, threshold, '>')]
            extract_rules(tree.children_right[node_id], right_conds)
        
        extract_rules(0, [])
        
        return candidates
    
    def deduplicate_candidates(self, candidates: list) -> list:
        """
        重複パターンを除去
        
        Args:
            candidates: パターン候補リスト
        
        Returns:
            list: 重複除去後のリスト
        """
        seen = set()
        unique = []
        
        for c in candidates:
            # 条件のハッシュを作成
            cond_key = json.dumps(c['pattern_conditions'], sort_keys=True)
            
            if cond_key not in seen:
                seen.add(cond_key)
                unique.append(c)
            else:
                # 既存のものより効果量が大きければ置換
                for i, existing in enumerate(unique):
                    existing_key = json.dumps(existing['pattern_conditions'], sort_keys=True)
                    if existing_key == cond_key and abs(c['effect_size']) > abs(existing['effect_size']):
                        unique[i] = c
                        break
        
        return unique
    
    def save_candidates(self, candidates: list) -> int:
        """
        パターン候補をDBに保存
        
        Args:
            candidates: パターン候補リスト
        
        Returns:
            int: 保存件数
        """
        saved_count = 0
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            for c in candidates:
                try:
                    cursor.execute("""
                        INSERT INTO pattern_candidates (
                            pattern_type, pattern_name, pattern_conditions,
                            extraction_method, extraction_date,
                            sample_size, effect_size, p_value,
                            validation_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """, (
                        c['pattern_type'],
                        c['pattern_name'],
                        json.dumps(c['pattern_conditions'], ensure_ascii=False),
                        c['extraction_method'],
                        datetime.now().strftime('%Y-%m-%d'),
                        c['sample_size'],
                        c['effect_size'],
                        c.get('p_value')
                    ))
                    saved_count += 1
                except Exception as e:
                    print(f"⚠️  保存エラー: {c['pattern_name']} - {e}")
            
            conn.commit()
        
        return saved_count
    
    def extract_all(
        self,
        start_date: str = None,
        end_date: str = None,
        save: bool = True
    ) -> dict:
        """
        全抽出方法を実行
        
        Args:
            start_date: 分析開始日
            end_date: 分析終了日
            save: DBに保存するか
        
        Returns:
            dict: 抽出結果
        """
        print("="*60)
        print("🔬 パターン抽出開始")
        print("="*60)
        
        # データ取得
        df = self.get_analysis_data(start_date, end_date)
        
        if df.empty:
            print("  分析データなし")
            return {
                'total_candidates': 0,
                'by_method': {},
                'saved': 0
            }
        
        print(f"  分析対象: {len(df)}レース")
        print(f"  期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")
        print(f"  的中率: {df['is_hit_1st'].mean()*100:.1f}%")
        print()
        
        all_candidates = []
        by_method = {}
        
        # 1. 頻度分析
        print("📊 頻度分析...")
        freq_candidates = self.extract_by_frequency(df)
        all_candidates.extend(freq_candidates)
        by_method['frequency'] = len(freq_candidates)
        print(f"  → {len(freq_candidates)}件")
        
        # 2. 統計比較
        print("📈 統計比較（カイ二乗検定）...")
        stat_candidates = self.extract_by_statistics(df)
        all_candidates.extend(stat_candidates)
        by_method['chi_square'] = len(stat_candidates)
        print(f"  → {len(stat_candidates)}件")
        
        # 3. 決定木
        print("🌳 決定木分析...")
        tree_candidates = self.extract_by_decision_tree(df)
        all_candidates.extend(tree_candidates)
        by_method['decision_tree'] = len(tree_candidates)
        print(f"  → {len(tree_candidates)}件")
        
        # 重複除去
        print()
        print("🧹 重複除去...")
        unique_candidates = self.deduplicate_candidates(all_candidates)
        print(f"  {len(all_candidates)} → {len(unique_candidates)}件")
        
        # 効果量でソート
        unique_candidates.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        # 保存
        saved_count = 0
        if save and unique_candidates:
            print()
            print("💾 DBに保存...")
            saved_count = self.save_candidates(unique_candidates)
            print(f"  → {saved_count}件保存")
        
        # 結果サマリー
        print()
        print("-"*60)
        print(f"✅ 抽出完了")
        print(f"  候補数: {len(unique_candidates)}件")
        print(f"  保存数: {saved_count}件")
        
        # トップ5を表示
        if unique_candidates:
            print()
            print("📋 効果量トップ5:")
            for i, c in enumerate(unique_candidates[:5]):
                sign = '+' if c['effect_size'] > 0 else ''
                print(f"  {i+1}. {c['pattern_name']}")
                print(f"     効果: {sign}{c['effect_size']:.1f}pt | サンプル: {c['sample_size']} | 方法: {c['extraction_method']}")
        
        print("="*60)
        
        self.candidates = unique_candidates
        
        return {
            'total_candidates': len(unique_candidates),
            'by_method': by_method,
            'saved': saved_count,
            'candidates': unique_candidates
        }


def test_extractor():
    """テスト関数"""
    print("="*60)
    print("🧪 PatternExtractor テスト")
    print("="*60)
    
    # テストデータを準備
    from src.learning.prediction_logger import PredictionLogger
    from src.learning.result_collector import ResultCollector
    from src.learning.result_analyzer import ResultAnalyzer
    
    # DBから複数のレースを取得してテスト
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT race_id, date
            FROM race_results
            WHERE date >= '2024-01-01' AND date <= '2024-01-31'
            ORDER BY date
            LIMIT 20
        """)
        races = cursor.fetchall()
    
    if len(races) < 5:
        print("❌ テスト用レースデータが不足しています")
        return
    
    print(f"  テスト用レース数: {len(races)}")
    
    logger = PredictionLogger(model_version="2.4-test")
    collector = ResultCollector()
    analyzer = ResultAnalyzer()
    
    # 各レースの予測データを作成・登録
    for race_id, race_date in races:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT horse_no, horse_name, odds_win, popularity, finish_position
                FROM race_results
                WHERE race_id = ?
                ORDER BY finish_position
                LIMIT 10
            """, (race_id,))
            horses = cursor.fetchall()
        
        if len(horses) < 3:
            continue
        
        # ランダムに予測順位を振る（テスト用）
        test_data = []
        for i, h in enumerate(horses):
            test_data.append({
                'race_id': race_id,
                'date': race_date,
                'place_code': race_id.split('_')[1] if '_' in race_id else '05',
                'place_name': '東京',
                'race_no': 1,
                'track_type': '芝' if i % 2 == 0 else 'ダート',
                'distance': 1600 + (i * 200),
                'track_condition': ['良', '稍重', '重', '不良'][i % 4],
                'weather': ['晴', '曇', '雨'][i % 3],
                'field_size': len(horses),
                'horse_no': h[0],
                'horse_name': h[1],
                'score': 90 - i * 5,
                'pred_rank': i + 1,
                'odds_win': h[2],
                'popularity': h[3]
            })
        
        test_df = pd.DataFrame(test_data)
        logger.log_predictions(test_df)
    
    # 結果収集
    print()
    collector.compare_and_save()
    
    # 差分分析
    print()
    analyzer.analyze_results()
    
    # パターン抽出
    print()
    extractor = PatternExtractor(min_sample_size=3, min_effect_size=1.0)  # テスト用に閾値を下げる
    result = extractor.extract_all(save=True)
    
    print()
    print(f"📊 テスト結果:")
    print(f"  抽出候補数: {result['total_candidates']}")
    print(f"  保存数: {result['saved']}")
    
    # pattern_candidates の内容確認
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pattern_name, pattern_type, extraction_method, effect_size, sample_size
            FROM pattern_candidates
            ORDER BY ABS(effect_size) DESC
            LIMIT 5
        """)
        rows = cursor.fetchall()
    
    if rows:
        print()
        print("📋 pattern_candidates 内容（トップ5）:")
        for row in rows:
            print(f"  {row[0]} | {row[1]} | {row[2]} | 効果{row[3]:.1f}pt | n={row[4]}")
    
    # クリーンアップ
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_logs WHERE model_version = '2.4-test'")
        cursor.execute("DELETE FROM prediction_results WHERE race_date LIKE '2024-01%'")
        cursor.execute("DELETE FROM pattern_candidates")
        conn.commit()
    print()
    print("  テストデータ削除完了")
    print()
    print("✅ テスト完了")


if __name__ == "__main__":
    test_extractor()


