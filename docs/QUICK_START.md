# Keiba AI v3 - クイックスタート

## 最終更新: 2025-12-08

## 現在の状態
- **Phase**: 自己学習システム構築
- **精度**: Top-1 37.99%, Top-3 65.63%（v2.5時点）
- **データ**: 2015-2025年 全レース

## 自己学習システム進捗

### ✅ 完了
- [x] 新プロジェクト構造作成
- [x] 基盤コード（config, database）
- [x] 自己学習スキーマ設計

### 🔄 進行中
- [ ] スキーマをDBに適用
- [ ] 予測ログ記録機能

### 📋 未着手
- [ ] 結果自動収集
- [ ] 差分分析エンジン
- [ ] パターン抽出
- [ ] バックテスト検証
- [ ] 自動登録

## 次のタスク
```powershell
cd C:\keiba-ai-v3
python src/learning/schema.py
```

## データベーステーブル
| テーブル | 用途 |
|----------|------|
| prediction_logs | 予測時の全情報 |
| prediction_results | 予測vs結果 |
| pattern_candidates | パターン候補 |
| validated_patterns | 検証済みパターン |
| pattern_performance | 性能監視 |
| learning_history | 学習履歴 |
