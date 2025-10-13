# 列名定義
"""
アプリケーションで使用する列名の定数定義
"""

# ユーザー入力項目（9列）
USER_INPUT_COLUMNS = [
    '年月日',
    '場所', 
    '芝・ダ',
    '距離',
    '馬場状態',
    '馬名',
    '馬番',
    '父馬名',
    '母の父馬名'
]

# 自動付与項目（4列）
BLOODLINE_COLUMNS = [
    '父馬名_小系統',
    '父馬名_国系統', 
    '母の父馬名_小系統',
    '母の父馬名_国系統'
]

# 予測用全項目（13列）
ALL_FEATURES = USER_INPUT_COLUMNS + BLOODLINE_COLUMNS

# 必須入力列（予測に最低限必要な列）
REQUIRED_COLUMNS = [
    '年月日',
    '場所',
    '芝・ダ',
    '距離',
    '馬場状態',
    '馬番'
]

# モデル学習時の特徴量名（エンコード済み列含む）
FEATURE_NAMES = [
    '距離', '馬番', '年', '月', '曜日', '半期',
    '場所_encoded', '芝・ダ_encoded', '馬場状態_encoded',
    '父馬名_小系統_encoded', '父馬名_国系統_encoded',
    '母の父馬名_小系統_encoded', '母の父馬名_国系統_encoded',
    '父母系統組合せ_encoded', '距離カテゴリ_encoded',
    '父血統有無', '母父血統有無'
]

# カテゴリカル特徴量
CATEGORICAL_FEATURES = [
    '場所', '芝・ダ', '馬場状態',
    '父馬名_小系統', '父馬名_国系統',
    '母の父馬名_小系統', '母の父馬名_国系統',
    '父母系統組合せ', '距離カテゴリ'
]

# 血統マスタの列名
BLOODLINE_MASTER_COLUMNS = {
    'horse_name': '馬名',
    'small_lineage': '小系統',
    'country_lineage': '国系統'
}

# 予測結果の列名
PREDICTION_COLUMN = 'predicted_time_sec'

# デフォルト値
DEFAULT_VALUES = {
    'unknown_bloodline': 'UNK',
    'empty_string': ''
}
