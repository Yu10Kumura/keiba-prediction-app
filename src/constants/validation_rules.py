# バリデーションルール定義
"""
入力データのバリデーションルール
"""

from datetime import datetime

# 必須項目
REQUIRED_COLUMNS = [
    '年月日', '場所', '芝・ダ', '距離', '馬場状態', 
    '馬名', '馬番', '父馬名', '母の父馬名'
]

# データ型制約
DATA_TYPE_RULES = {
    '年月日': 'date',
    '場所': 'string',
    '芝・ダ': 'string', 
    '距離': 'int',
    '馬場状態': 'string',
    '馬名': 'string',
    '馬番': 'int',
    '父馬名': 'string',
    '母の父馬名': 'string'
}

# 値の制約
VALUE_CONSTRAINTS = {
    '年月日': {
        'min_date': datetime(1990, 1, 1),
        'max_date': datetime(2030, 12, 31)
    },
    '距離': {
        'min_value': 800,
        'max_value': 4000
    },
    '馬番': {
        'min_value': 1,
        'max_value': 18
    },
    '芝・ダ': {
        'allowed_values': ['芝', 'ダ']
    },
    '馬場状態': {
        'allowed_values': ['良', '稍重', '重', '不良']
    }
}

# 文字列長制約
STRING_LENGTH_CONSTRAINTS = {
    '馬名': {'max_length': 30},
    '父馬名': {'max_length': 30}, 
    '母の父馬名': {'max_length': 30}
}

# 競馬場リスト（主要競馬場）
VALID_VENUES = [
    '札幌', '函館', '福島', '新潟', '東京', '中山', '中京', 
    '京都', '阪神', '小倉', '大井', '川崎', '船橋', '浦和',
    '水沢', '盛岡', '門別', '帯広', '旭川', '佐賀', '荒尾',
    '高知', '園田', '姫路', '益田', '笠松', '名古屋', '金沢'
]

# ファイル制約
FILE_CONSTRAINTS = {
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'max_rows': 10000,
    'allowed_extensions': ['.csv', '.xlsx'],
    'allowed_encodings': ['utf-8', 'utf-8-sig', 'shift_jis']
}

# 統合バリデーションルール
VALIDATION_RULES = {
    '年月日': {
        'required': True,
        'dtype': 'str',
        'min_date': datetime(1990, 1, 1),
        'max_date': datetime(2030, 12, 31)
    },
    '場所': {
        'required': True,
        'dtype': 'str',
        'allowed_values': VALID_VENUES
    },
    '芝・ダ': {
        'required': True,
        'dtype': 'str',
        'allowed_values': ['芝', 'ダ']
    },
    '距離': {
        'required': True,
        'dtype': 'int',
        'min_value': 800,
        'max_value': 4000
    },
    '馬場状態': {
        'required': True,
        'dtype': 'str',
        'allowed_values': ['良', '稍重', '重', '不良']
    },
    '馬名': {
        'required': False,
        'dtype': 'str',
        'max_length': 30
    },
    '馬番': {
        'required': True,
        'dtype': 'int',
        'min_value': 1,
        'max_value': 18
    },
    '父馬名': {
        'required': False,
        'dtype': 'str',
        'max_length': 30
    },
    '母の父馬名': {
        'required': False,
        'dtype': 'str',
        'max_length': 30
    },
    '走破タイム': {
        'required': False,
        'dtype': 'float',
        'min_value': 0.1,
        'max_value': 600.0
    }
}
