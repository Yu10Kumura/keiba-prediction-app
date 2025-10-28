# 🚀 競馬予測システム クイックスタートプロンプト

## 基本版（簡単な作業用）

```
競馬予測システムの作業を再開したいです。

システム: /Users/yutokumura/Desktop/競馬Sim/Keiba_Python/ver1_streamlit_sim_app/
URL: https://keiba-prediction-app-sadakai.streamlit.app/
状況: CSV無限ループ・馬名表示の問題は解決済み

やりたいこと: [具体的な作業内容]
```

## 詳細版（複雑な作業用）

```
競馬の走破タイム予測システムについて作業したいです。

【システム概要】
- LightGBMを使った機械学習による競馬走破タイム予測システム
- Streamlit Webアプリケーション
- CSVファイルアップロード + 手動入力対応
- 11頭のバッチ予測機能

【現在の状況】
- ✅ Streamlit Cloud デプロイ完了
- ✅ CSV無限ループ問題 解決済み
- ✅ 馬名表示機能 修正完了
- ✅ 本番環境で正常動作確認済み

【技術情報】
- プロジェクト場所: /Users/yutokumura/Desktop/競馬Sim/Keiba_Python/ver1_streamlit_sim_app/
- GitHub: https://github.com/Yu10Kumura/keiba-prediction-app
- 本番URL: https://keiba-prediction-app-sadakai.streamlit.app/
- 主要言語: Python, Streamlit, LightGBM
- 開発環境: Python 3.11+, VS Code

【重要ファイル】
- app.py: メインアプリケーション
- src/components/data_input.py: CSV処理（無限ループ修正済み）
- src/components/result_display.py: 結果表示（馬名取得修正済み）
- DEVELOPMENT_LOG.md: 詳細な開発履歴とトラブルシューティング

【最近の主要修正】
1. CSV無限ループ修正（st.rerun()問題解決）
2. 馬名表示修正（原CSVのG列から取得）
3. セッション状態管理の改善

【今やりたいこと】
[ここに具体的な作業内容を記載]
例：新機能追加 / バグ修正 / UI改善 / 性能向上 など

まず現在のシステム状況を確認してから、作業を開始してください。
```

## 🔧 デバッグ・トラブルシューティング版

```
競馬予測システムでエラーが発生しています。

システム: /Users/yutokumura/Desktop/競馬Sim/Keiba_Python/ver1_streamlit_sim_app/
エラー状況: [具体的なエラー内容]

過去に解決済みの問題:
- CSV無限ループ（st.rerun()削除で解決）
- 馬名表示問題（original_csv_data保持で解決）

DEVELOPMENT_LOG.mdのトラブルシューティングセクションも参照してください。
```

## 📋 新機能開発版

```
競馬予測システムに新機能を追加したいです。

現在のシステム: /Users/yutokumura/Desktop/競馬Sim/Keiba_Python/ver1_streamlit_sim_app/
現在の機能: CSV/手動入力、LightGBM予測、11頭バッチ処理、馬名表示

追加したい機能: [具体的な新機能]

既存機能への影響を最小限に抑えて実装してください。
DEVELOPMENT_LOG.mdで現在の技術構成を確認してから開始してください。
```

## 🎯 使い分けガイド

| 作業内容 | 使用するプロンプト |
|---------|-------------------|
| 簡単なバグ修正 | 基本版 |
| 新機能追加 | 新機能開発版 |
| 複雑な修正・改善 | 詳細版 |
| エラー解決 | デバッグ版 |

## 💡 効果的な使用のコツ

1. **作業内容を具体的に記載**
   - ❌ 「改善したい」
   - ✅ 「予測結果をグラフで可視化したい」

2. **エラーは詳細情報も追加**
   - エラーメッセージ
   - 発生条件
   - 期待する動作

3. **DEVELOPMENT_LOG.mdと併用**
   - 技術詳細はログファイルを参照
   - プロンプトは作業開始用

## 🔗 関連ファイル

- `DEVELOPMENT_LOG.md`: 詳細な開発履歴
- `README.md`: プロジェクト概要
- `requirements.txt`: 依存関係