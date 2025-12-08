# Keiba AI v3 - 自己学習型競馬予測システム

## 概要
完全自動の自己学習機能を搭載した競馬予測AI

## 自己学習サイクル
1. 予測実行 → prediction_logs に記録
2. 結果収集 → prediction_results に記録  
3. 差分分析 → pattern_candidates に候補抽出
4. 検証 → validated_patterns に登録
5. 次回予測に適用
