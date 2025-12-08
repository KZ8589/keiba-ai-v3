# データベース クイックリファレンス

**最終更新**: 2025-11-28

---

## 🚀 最も重要な3つのルール

1. **コード実装前にスキーマ確認** - 絶対に推測しない
2. **正しいカラム名を使う** - `race_no` ✅ / `race_number` ❌
3. **race_idの形式** - `2025-11-09_03_03_02_01` ✅

---

## 📊 主要テーブル

### race_results（馬別結果）- 64カラム
```sql
-- 基本情報
race_id          -- 2025-11-09_03_03_02_01
date             -- 2025-11-09
place_code       -- 03 (福島)
race_no          -- 01
horse_id         -- 2023100032
horse_no         -- 馬番（1-18）
finish_position  -- 1 (着順)

-- 馬情報
horse_name       -- 馬名
horse_age        -- 馬齢（2-10）
horse_sex        -- 性別（牡/牝/セ）
horse_weight     -- 馬体重（400-550）
gate_no          -- 枠番（1-8）

-- 騎手・調教師
jockey_id        -- 騎手ID
jockey_name      -- 騎手名
trainer_id       -- 調教師ID
trainer_name     -- 調教師名

-- オッズ・人気
odds_win         -- 単勝オッズ（13.5）
popularity       -- 人気順（1-18）

-- レース結果
time             -- 走破タイム（秒）
last_3f_time     -- 上がり3Fタイム（秒）
corner1-4        -- 通過順位
margin_time      -- 着差タイム
load_weight      -- 斤量（55.0）

-- 配当データ
win_payout           -- 単勝配当
place_payout         -- 複勝配当
wakuren_payout       -- 枠連配当
umaren_payout        -- 馬連配当
umatan_payout        -- 馬単配当
sanrenpuku_payout    -- 3連複配当
sanrentan_payout     -- 3連単配当

-- 天気・馬場（補完済み）
weather              -- 晴/曇/雨/小雨/雪/小雪
track_condition_csv2 -- 良/稍重/重/不良

-- メタ情報
field_size       -- 出走頭数
created_at       -- 作成日時
data_type        -- データソース
```

### race_details（レース詳細）- 32カラム
```sql
race_id           -- 主キー
place_name        -- 福島
distance          -- 2000
track_type        -- 芝 or ダート
track_condition   -- 良/稍重/重/不良
weather           -- 晴/曇/雨
race_class        -- レースクラス
```

---

## 📈 カラム充填率

| カラム | 充填率 | 備考 |
|--------|--------|------|
| 基本情報 | 100% | ✅ |
| horse_age | 100% | ✅ |
| horse_sex | 100% | ✅ |
| load_weight | 100% | ✅ |
| gate_no | 100% | ✅ |
| last_3f_time | 100% | ✅ |
| **weather** | **100%** | ✅ 補完完了 |
| win_payout | 95.4% | ✅ |
| place_payout | 95.4% | ✅ |
| wakuren_payout | 89.8% | ✅ |
| umaren_payout | 95.4% | ✅ |
| umatan_payout | 95.4% | ✅ |
| sanrenpuku_payout | 95.4% | ✅ |
| sanrentan_payout | 95.4% | ✅ |

---

## 🌤️ 天気データ分布

| 天気 | 件数 | 割合 |
|------|------|------|
| 晴 | 79,067件 | 42.6% |
| 曇 | 43,881件 | 23.7% |
| 小雨 | 6,647件 | 3.6% |
| 雨 | 6,488件 | 3.5% |
| 雪 | 131件 | 0.1% |
| 小雪 | 72件 | 0.04% |

---

## 🔑 場所コード

| 場所 | コード | CSV表記 |
|-----|-------|---------|
| 札幌 | 01 | 札 |
| 函館 | 02 | 函 |
| 福島 | 03 | 福 |
| 新潟 | 04 | 新 |
| 東京 | 05 | 東 |
| **中山** | **06** | **中** |
| **中京** | **07** | **名** |
| 京都 | 08 | 京 |
| 阪神 | 09 | 阪 |
| 小倉 | 10 | 小 |

⚠️ **重要**: CSVで「中」は中山(06)、「名」は中京(07)

---

## ⚠️ よくある間違い

| ❌ 間違い | ✅ 正しい |
|----------|---------|
| `race_number` | `race_no` |
| `place` | `place_code` |
| `weight` | `horse_weight` |
| `position` | `finish_position` |
| `bracket_no` | `gate_no` |
| `impost` | `load_weight` |
| `trifecta_payout` | `sanrenpuku_payout` |

---

## 🛠️ スキーマ確認コマンド
```python
import sqlite3
conn = sqlite3.connect('data/keiba.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(race_results)")
for col in cursor.fetchall():
    print(f"{col[1]}")
conn.close()
```

---

詳細は `DATABASE_SCHEMA_REFERENCE.md` を参照
