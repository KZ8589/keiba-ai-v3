"""
config.py - 自己学習システム設定の一元管理

すべてのビン定義、閾値、設定をここで管理
"""
from pathlib import Path

# ============================================================
# パス設定
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'keiba.db'
CSV_DIR = PROJECT_ROOT / 'data' / 'csv_imports'

# ============================================================
# 数値特徴量のビン定義
# 抽出・検証・適用で統一して使用
# ============================================================
FEATURE_BINS = {
    'odds_win': {
        'bins': [0, 3, 10, 30, float('inf')],
        'labels': ['低オッズ', '中オッズ', '高オッズ', '超高オッズ']
    },
    'popularity': {
        'bins': [0, 3, 6, float('inf')],
        'labels': ['上位人気', '中位人気', '下位人気']
    },
    'field_size': {
        'bins': [0, 10, 14, float('inf')],
        'labels': ['少頭数', '標準頭数', '多頭数']
    },
    'distance': {
        'bins': [0, 1400, 1800, 2200, float('inf')],
        'labels': ['短距離', 'マイル', '中距離', '長距離']
    },
    'horse_age': {
        'bins': [0, 3, 5, float('inf')],
        'labels': ['若馬', '中堅', '古馬']
    }
}

# カテゴリカル特徴量（そのまま使用）
CATEGORICAL_FEATURES = [
    'track_type',       # 芝/ダート
    'track_condition',  # 良/稀重/重/不良
    'weather',          # 晴/曇/雨/etc
    'place_code',       # 01-10
    'running_style'     # 逃げ/先行/中団/後方
]

# ============================================================
# パターン抽出設定
# ============================================================
EXTRACTION_CONFIG = {
    'min_sample_size': 100,      # 最小サンプルサイズ
    'min_effect_size': 3.0,      # 最小効果量（ポイント差）
    'min_p_value': 0.05,         # p値閾値
}

# ============================================================
# パターン検証設定
# ============================================================
VALIDATION_CONFIG = {
    'min_sample_size': 500,      # バックテスト最小サンプル
    'min_effect_size': 1.0,      # 最小効果量
    'min_p_value': 0.05,         # p値閾値
    'min_validation_periods': 5, # 最小検証期間数（四半期）
    'min_consistency': 0.6,      # 一貫性閾値
}

# ============================================================
# パターン適用設定
# ============================================================
APPLICATION_CONFIG = {
    'max_adjustment': 10.0,      # 最大スコア調整幅
    'min_adjustment': -10.0,     # 最小スコア調整幅
}

# ============================================================
# ユーティリティ関数
# ============================================================
def get_bin_label(feature: str, value: float) -> str:
    """
    数値を対応するビンラベルに変換
    
    Args:
        feature: 特徴量名
        value: 数値
    
    Returns:
        str: ビンラベル（該当なしは'unknown'）
    """
    if feature not in FEATURE_BINS:
        return 'unknown'
    
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return 'unknown'
    
    bins = FEATURE_BINS[feature]['bins']
    labels = FEATURE_BINS[feature]['labels']
    
    for i in range(len(bins) - 1):
        if bins[i] < value <= bins[i + 1]:
            return labels[i]
    
    # 最小値以下の場合は最初のラベル
    if value <= bins[0]:
        return labels[0]
    
    return 'unknown'


def get_bin_range(feature: str, label: str) -> tuple:
    """
    ビンラベルを数値範囲に変換
    
    Args:
        feature: 特徴量名
        label: ビンラベル
    
    Returns:
        tuple: (min, max) または None
    """
    if feature not in FEATURE_BINS:
        return None
    
    bins = FEATURE_BINS[feature]['bins']
    labels = FEATURE_BINS[feature]['labels']
    
    if label not in labels:
        return None
    
    idx = labels.index(label)
    return (bins[idx], bins[idx + 1])


def is_in_bin(feature: str, value: float, label: str) -> bool:
    """
    数値が指定ビンに含まれるか判定
    
    Args:
        feature: 特徴量名
        value: 数値
        label: ビンラベル
    
    Returns:
        bool: 含まれればTrue
    """
    range_tuple = get_bin_range(feature, label)
    if range_tuple is None:
        return False
    
    return range_tuple[0] < value <= range_tuple[1]


# ============================================================
# テスト
# ============================================================
if __name__ == "__main__":
    print("="*50)
    print("🧪 config.py テスト")
    print("="*50)
    
    # ビンラベル変換テスト
    test_cases = [
        ('odds_win', 2.5, '低オッズ'),
        ('odds_win', 8.0, '中オッズ'),
        ('odds_win', 25.0, '高オッズ'),
        ('odds_win', 50.0, '超高オッズ'),
        ('popularity', 2, '上位人気'),
        ('popularity', 5, '中位人気'),
        ('distance', 1200, '短距離'),
        ('distance', 1600, 'マイル'),
        ('distance', 2000, '中距離'),
    ]
    
    print("\n📊 get_bin_label() テスト:")
    for feature, value, expected in test_cases:
        result = get_bin_label(feature, value)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {feature}={value} → {result} (期待: {expected})")
    
    print("\n📊 get_bin_range() テスト:")
    for feature in ['odds_win', 'popularity']:
        for label in FEATURE_BINS[feature]['labels']:
            range_tuple = get_bin_range(feature, label)
            print(f"  {feature}/{label} → {range_tuple}")
    
    print("\n✅ テスト完了")
