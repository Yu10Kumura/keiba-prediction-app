# 競馬予測システム 再開用プロンプト

## 🎯 新しいチャットでこのプロンプトをコピペして使用してください

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

## 📋 使用方法

1. **新しいチャットを開始**
2. **上記のプロンプトをコピペ**
3. **「今やりたいこと」部分を具体的に記載**
4. **DEVELOPMENT_LOG.mdも一緒に参照**

## 🎯 プロンプトのポイント

- **現在の完成状態を明記**（作業済み項目）
- **重要な技術情報を整理**（パス、URL、技術スタック）
- **解決済み問題を明確化**（同じ問題の再発防止）
- **すぐに作業開始できる情報量**（コンテキスト設定時間短縮）