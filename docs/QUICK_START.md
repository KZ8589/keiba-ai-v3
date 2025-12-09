# Keiba AI v3 - クイックスタート

## 最終更新: 2025-12-09

## 現在の状態
- **Phase**: 自己学習システム構築
- **精度**: Top-1 37.99%, Top-3 65.63%（v2.5時点）
- **データ**: 2015-2025年 全レース（keiba.db 1.4GB）

## 本日の成果（2025-12-09）
- [x] 新プロジェクト keiba-ai-v3 作成
- [x] 基盤コード（config.py, database.py）
- [x] 自己学習スキーマ（6テーブル）作成・適用
- [x] GitHub連携（https://github.com/KZ8589/keiba-ai-v3）
- [x] Claude Project連携（12ファイル）

## 自己学習テーブル（作成済み）
| テーブル | 用途 |
|----------|------|
| prediction_logs | 予測時の全情報記録 |
| prediction_results | 予測vs結果比較 |
| pattern_candidates | パターン候補 |
| validated_patterns | 検証済みパターン |
| pattern_performance | 性能監視 |
| learning_history | 学習履歴 |

## 次のタスク
1. prediction_logger.py - 予測ログ記録機能
2. result_collector.py - 結果自動収集
3. result_analyzer.py - 差分分析エンジン
4. pattern_extractor.py - パターン抽出
5. pattern_validator.py - バックテスト検証

## 環境
```powershell
cd C:\keiba-ai-v3
.\venv64\Scripts\Activate.ps1
```

## Git操作
```powershell
git status
git add .
git commit -m "feat: 機能名"
git push
```
