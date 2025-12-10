"""
learning_pipeline.py - 自己学習パイプライン統合オーケストレーター

フロー:
1. run_weekly_prediction() - 週末予測実行→ログ記録
2. run_result_collection() - レース終了後の結果収集→分析
3. run_pattern_learning() - パターン抽出→検証→登録
4. run_full_cycle() - 全サイクル実行
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from learning.prediction_logger import PredictionLogger
from learning.result_collector import ResultCollector
from learning.result_analyzer import ResultAnalyzer
from learning.pattern_extractor import PatternExtractor
from learning.pattern_validator import PatternValidator


class LearningPipeline:
    """自己学習パイプライン統合クラス"""
    
    def __init__(self, model_version: str = "3.0"):
        self.model_version = model_version
        self.logger = PredictionLogger(model_version=model_version)
        self.collector = ResultCollector()
        self.analyzer = ResultAnalyzer()
        self.extractor = PatternExtractor(min_sample_size=50, min_effect_size=2.0)
        self.validator = PatternValidator()
    
    def run_result_collection(self, race_date: str = None) -> dict:
        """
        結果収集フェーズ
        - 予測済みレースの実結果を収集
        - 予測と結果を比較して保存
        """
        print("="*60)
        print("📥 Phase 1: 結果収集")
        print("="*60)
        
        # 未収集のレースを取得
        pending_races = self.collector.get_pending_races(race_date)
        
        if not pending_races:
            print("  → 収集待ちレースなし")
            return {'collected': 0, 'races': []}
        
        print(f"  → 収集対象: {len(pending_races)}レース")
        
        # 結果収集・比較
        result = self.collector.compare_and_save(race_ids=pending_races, race_date=race_date)
        
        print(f"  → 収集完了: {result.get('saved', 0)}件")
        
        # 精度サマリー
        accuracy = self.collector.get_accuracy_summary(race_date)
        if accuracy.get('total_races', 0) > 0:
            print(f"  → 1着的中率: {accuracy.get('hit_rate_1st', 0):.1%}")
            print(f"  → Top3的中率: {accuracy.get('hit_rate_top3', 0):.1%}")
        
        return result
    
    def run_analysis(self, race_date: str = None) -> dict:
        """
        分析フェーズ
        - 予測と結果の差分を分析
        - 波乱度・失敗要因を特定
        """
        print("\n" + "="*60)
        print("🔍 Phase 2: 差分分析")
        print("="*60)
        
        # 未分析のレースを取得
        unanalyzed = self.analyzer.get_unanalyzed_results(race_date)
        
        if not unanalyzed:
            print("  → 分析待ちレースなし")
            return {'analyzed': 0}
        
        print(f"  → 分析対象: {len(unanalyzed)}レース")
        
        # 分析実行
        result = self.analyzer.analyze_results(race_ids=unanalyzed, race_date=race_date)
        
        print(f"  → 分析完了: {result.get('analyzed', 0)}件")
        
        # 波乱度サマリー
        summary = self.analyzer.get_analysis_summary(race_date)
        if summary.get('total', 0) > 0:
            print(f"  → 波乱なし: {summary.get('none', 0)}件")
            print(f"  → 軽度波乱: {summary.get('minor', 0)}件")
            print(f"  → 中度波乱: {summary.get('major', 0)}件")
            print(f"  → 大波乱: {summary.get('extreme', 0)}件")
        
        return result
    
    def run_pattern_extraction(self, start_date: str = None, end_date: str = None) -> dict:
        """
        パターン抽出フェーズ
        - 成功/失敗パターンを自動抽出
        - 候補をDBに保存
        """
        print("\n" + "="*60)
        print("🔎 Phase 3: パターン抽出")
        print("="*60)
        
        # 分析データ取得
        df = self.extractor.get_analysis_data(start_date, end_date)
        
        if df.empty or len(df) < 100:
            print(f"  → データ不足: {len(df)}件（最低100件必要）")
            return {'extracted': 0}
        
        print(f"  → 分析データ: {len(df)}件")
        
        # パターン抽出
        result = self.extractor.extract_all(start_date, end_date)
        
        print(f"  → 抽出完了: {result.get('total_candidates', 0)}件")
        print(f"    - 頻度分析: {result.get('frequency_patterns', 0)}件")
        print(f"    - 統計比較: {result.get('statistical_patterns', 0)}件")
        print(f"    - 決定木: {result.get('tree_patterns', 0)}件")
        
        return result
    
    def run_pattern_validation(self, start_date: str = None, end_date: str = None) -> dict:
        """
        パターン検証フェーズ
        - 候補パターンをバックテスト
        - 合格パターンを validated_patterns に登録
        """
        print("\n" + "="*60)
        print("✅ Phase 4: パターン検証")
        print("="*60)
        
        # 未検証の候補を取得
        pending = self.validator.get_pending_candidates()
        
        if not pending:
            print("  → 検証待ち候補なし")
            return {'validated': 0, 'rejected': 0}
        
        print(f"  → 検証対象: {len(pending)}件")
        
        # 検証実行
        result = self.validator.validate_all(start_date, end_date)
        
        print(f"  → 検証完了")
        print(f"    - 合格: {result.get('validated', 0)}件")
        print(f"    - 却下: {result.get('rejected', 0)}件")
        
        return result
    
    def run_post_race_cycle(self, race_date: str = None) -> dict:
        """
        レース終了後の学習サイクル
        1. 結果収集
        2. 差分分析
        3. パターン抽出
        4. パターン検証
        """
        print("\n" + "="*60)
        print("🔄 自己学習サイクル開始")
        print(f"   対象日: {race_date or '全日'}")
        print("="*60)
        
        results = {}
        
        # Phase 1: 結果収集
        results['collection'] = self.run_result_collection(race_date)
        
        # Phase 2: 差分分析
        results['analysis'] = self.run_analysis(race_date)
        
        # Phase 3: パターン抽出（十分なデータがある場合）
        results['extraction'] = self.run_pattern_extraction()
        
        # Phase 4: パターン検証
        results['validation'] = self.run_pattern_validation()
        
        # サマリー
        print("\n" + "="*60)
        print("📊 サイクル完了サマリー")
        print("="*60)
        print(f"  結果収集: {results['collection'].get('saved', 0)}件")
        print(f"  差分分析: {results['analysis'].get('analyzed', 0)}件")
        print(f"  パターン抽出: {results['extraction'].get('total_candidates', 0)}件")
        print(f"  パターン検証: 合格{results['validation'].get('validated', 0)}件 / 却下{results['validation'].get('rejected', 0)}件")
        
        return results
    
    def get_system_status(self) -> dict:
        """システム状態を取得"""
        import sqlite3
        from core.database import DB_PATH
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        status = {}
        
        # 各テーブルの件数
        tables = [
            'prediction_logs',
            'prediction_results', 
            'pattern_candidates',
            'validated_patterns'
        ]
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                status[table] = cursor.fetchone()[0]
            except:
                status[table] = 0
        
        # 未処理件数
        cursor.execute("""
            SELECT COUNT(*) FROM prediction_logs 
            WHERE race_id NOT IN (SELECT race_id FROM prediction_results)
        """)
        status['pending_collection'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM prediction_results 
            WHERE analyzed_at IS NULL
        """)
        status['pending_analysis'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM pattern_candidates 
            WHERE validation_status = 'pending'
        """)
        status['pending_validation'] = cursor.fetchone()[0]
        
        conn.close()
        
        return status
    
    def show_status(self):
        """システム状態を表示"""
        print("\n" + "="*60)
        print("📈 自己学習システム状態")
        print("="*60)
        
        status = self.get_system_status()
        
        print(f"\n📁 データ件数:")
        print(f"  予測ログ: {status.get('prediction_logs', 0):,}件")
        print(f"  予測結果: {status.get('prediction_results', 0):,}件")
        print(f"  パターン候補: {status.get('pattern_candidates', 0):,}件")
        print(f"  検証済パターン: {status.get('validated_patterns', 0):,}件")
        
        print(f"\n⏳ 未処理:")
        print(f"  結果収集待ち: {status.get('pending_collection', 0):,}件")
        print(f"  分析待ち: {status.get('pending_analysis', 0):,}件")
        print(f"  検証待ち: {status.get('pending_validation', 0):,}件")


def test_pipeline():
    """パイプラインテスト"""
    print("="*60)
    print("🧪 LearningPipeline テスト")
    print("="*60)
    
    pipeline = LearningPipeline(model_version="3.0")
    
    # システム状態表示
    pipeline.show_status()
    
    print("\n✅ パイプライン初期化成功")


if __name__ == "__main__":
    test_pipeline()


