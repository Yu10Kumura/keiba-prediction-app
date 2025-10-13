# フォルダ構成設計書

## プロジェクト構成

```
ver1.Streamlit.Sim.APP/
│
├── README.md                          # プロジェクト概要・セットアップ手順
├── requirements.txt                   # 依存ライブラリ
├── .streamlit/                        # Streamlit設定
│   └── config.toml                   # Streamlit設定ファイル
│
├── docs/                              # ドキュメント
│   ├── requirements_specification.md  # 要件定義書
│   ├── design_document.md            # 設計書
│   ├── user_manual.md                # ユーザーマニュアル
│   └── api_reference.md              # API仕様書（将来拡張）
│
├── src/                               # ソースコード
│   ├── app.py                        # Streamlitメインアプリ
│   ├── components/                   # UIコンポーネント
│   │   ├── __init__.py
│   │   ├── file_upload.py           # ファイルアップロード画面
│   │   ├── manual_input.py          # 手入力画面
│   │   ├── result_display.py        # 結果表示
│   │   └── sidebar.py               # サイドバー
│   │
│   ├── core/                         # コアロジック
│   │   ├── __init__.py
│   │   ├── predictor.py             # 予測エンジン
│   │   ├── data_processor.py        # データ前処理
│   │   ├── validator.py             # バリデーション
│   │   └── bloodline_manager.py     # 血統マスタ管理
│   │
│   ├── utils/                        # ユーティリティ
│   │   ├── __init__.py
│   │   ├── file_handler.py          # ファイル操作
│   │   ├── logger.py                # ログ管理
│   │   ├── config_manager.py        # 設定管理
│   │   └── error_handler.py         # エラーハンドリング
│   │
│   └── constants/                    # 定数定義
│       ├── __init__.py
│       ├── columns.py               # 列名定義
│       ├── validation_rules.py      # バリデーションルール
│       └── messages.py              # メッセージ定義
│
├── models/                           # 学習済みモデル
│   ├── lgb_model_final.pkl          # LightGBMモデル
│   ├── label_encoders.pkl           # ラベルエンコーダー
│   ├── feature_list.pkl             # 特徴量リスト
│   └── model_info.json             # モデル情報・メタデータ
│
├── data/                             # データファイル
│   ├── bloodline_master.csv         # 血統マスタ
│   ├── venue_master.csv             # 競馬場マスタ
│   ├── sample_input.xlsx            # サンプル入力ファイル
│   └── sample_input.csv             # サンプル入力ファイル（CSV版）
│
├── config/                           # 設定ファイル
│   ├── app_config.yaml              # アプリケーション設定
│   ├── validation_config.yaml       # バリデーション設定
│   └── model_config.yaml            # モデル設定
│
├── logs/                             # ログファイル
│   ├── app.log                      # アプリケーションログ
│   ├── error.log                    # エラーログ
│   └── access.log                   # アクセスログ（将来拡張）
│
├── tests/                            # テストコード
│   ├── __init__.py
│   ├── test_predictor.py            # 予測エンジンテスト
│   ├── test_validator.py            # バリデーションテスト
│   ├── test_data_processor.py       # データ処理テスト
│   └── test_bloodline_manager.py    # 血統マスタテスト
│
└── temp/                             # 一時ファイル
    └── .gitkeep                     # Git管理用（空フォルダ保持）
```

## 各ディレクトリの役割

### `/src/` - ソースコード
- **app.py**: Streamlitのメインアプリケーション
- **components/**: 画面単位のUIコンポーネント
- **core/**: ビジネスロジック・予測処理の中核
- **utils/**: 汎用的なユーティリティ機能
- **constants/**: 定数・設定値の一元管理

### `/models/` - 学習済みモデル
- 既存のml_modelsから必要ファイルをコピー
- モデルファイル + メタデータで管理

### `/data/` - マスタデータ
- 血統マスタ、競馬場マスタ等の参照データ
- サンプルファイルでユーザー支援

### `/config/` - 設定管理
- YAML形式での設定管理
- 環境別設定の切り替え対応

### `/docs/` - ドキュメント
- 要件定義、設計書、マニュアル類
- 開発・運用に必要な文書一式

### `/tests/` - テストコード
- 単体テスト、結合テストの実装
- CI/CD対応の準備

### `/logs/` - ログ管理
- アプリケーション実行ログ
- エラーログ、アクセスログ（将来拡張）

## ファイル命名規則

### Python ファイル
- **snake_case**: `data_processor.py`
- **クラス名**: PascalCase `DataProcessor`
- **関数名**: snake_case `process_data()`

### 設定ファイル
- **YAML**: `app_config.yaml`
- **環境別**: `app_config_dev.yaml`

### データファイル
- **マスタ**: `{テーブル名}_master.csv`
- **サンプル**: `sample_{用途}.{拡張子}`

## モジュール分割方針

### 責任の分離
- **UI層**: Streamlitコンポーネント（components/）
- **ビジネス層**: 予測・検証ロジック（core/）
- **インフラ層**: ファイル操作・ログ等（utils/）

### 再利用性
- 共通処理はutils/で一元化
- 設定値はconstants/で管理
- モジュール間の依存関係を最小化

### テスタビリティ
- 各モジュールの単体テスト実装
- モックを活用した独立テスト
- 設定外部化によるテスト容易性

## 導入・展開手順

### 1. 環境準備
```bash
cd ver1.Streamlit.Sim.APP/
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. モデルファイルコピー
```bash
# 既存ml_modelsからコピー
cp ../ml_models/models/lgb_model_final.pkl models/
cp ../ml_models/models/*.pkl models/
```

### 3. 設定ファイル準備
- config/内のYAMLファイル編集
- 血統マスタ等のデータファイル配置

### 4. アプリケーション起動
```bash
streamlit run src/app.py
```

## 今後の拡張対応

### スケーラビリティ
- データベース導入時: data/層をDB接続に変更
- API化時: FastAPI等との連携層追加
- マイクロサービス化: core/を独立サービス化

### 運用性
- Docker化: Dockerfile追加
- CI/CD: GitHub Actions等の設定追加
- 監視: Prometheus/Grafana等との連携

このフォルダ構成により、開発・運用・拡張のすべてのフェーズで効率的な管理が可能になります。
