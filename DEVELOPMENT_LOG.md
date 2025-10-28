# 競馬予測システム 開発ログ

**作成日：** 2025年10月28日  
**プロジェクト：** 競馬走破タイム予測システム  
**技術スタック：** Python, Streamlit, LightGBM, GitHub, Streamlit Cloud

---

## 📋 プロジェクト概要

### システム仕様
- **目的：** 機械学習を使用した競馬の走破タイム予測
- **機能：** CSVファイルアップロード、手動入力、バッチ予測
- **デプロイ：** Streamlit Cloud（本番環境）
- **リポジトリ：** https://github.com/Yu10Kumura/keiba-prediction-app
- **本番URL：** https://keiba-prediction-app-sadakai.streamlit.app/

### 技術構成
```
├── app.py                 # メインアプリケーション
├── src/
│   ├── components/
│   │   ├── data_input.py      # CSVアップロード処理
│   │   ├── manual_input.py    # 手動入力フォーム
│   │   └── result_display.py  # 予測結果表示
│   ├── core/
│   │   ├── prediction_engine.py  # LightGBM予測エンジン
│   │   ├── data_processor.py     # データ前処理
│   │   └── bloodline_manager.py  # 血統分類管理
│   └── utils/
└── models/                # 学習済みモデル
```

---

## 🐛 解決した問題とその対策

### 1. CSV無限ループ問題（最重要）

**問題：**
- CSVファイルアップロード時に無限ループが発生
- ページが応答しなくなる

**原因：**
- `st.rerun()`の過度な呼び出し
- セッション状態管理の複雑化

**解決策：**
```python
# data_input.py の修正
def render_data_processing(self, uploaded_data):
    # st.rerun()呼び出しを削除
    # セッション状態管理をシンプル化
    processed_data = self.data_processor.process_data(uploaded_data)
    return processed_data, processing_info
```

**修正ファイル：** `src/components/data_input.py`

### 2. 馬名表示問題

**問題：**
- 予測結果で馬名が「馬1」「馬2」と表示される
- 実際のCSVの馬名が反映されない

**原因：**
- データ処理パイプラインで元のCSV情報が失われる
- 処理済みデータに馬名情報が含まれない

**解決策：**
```python
# app.py での原データ保持
if uploaded_data is not None:
    # Store original CSV data for horse names
    st.session_state.original_csv_data = uploaded_data

# result_display.py での馬名取得
def render_prediction_results(self, predictions, input_data):
    # G列（インデックス6）から馬名を直接取得
    if input_data is not None and input_data.shape[1] > 6:
        horse_names = input_data.iloc[:, 6].astype(str).tolist()
```

**修正ファイル：** `app.py`, `src/components/result_display.py`

### 3. 予測結果表示の改善

**問題：**
- 単一の予測結果しか表示されない
- 11頭すべての結果が見えない

**解決策：**
- バッチ処理対応
- 全馬の予測結果をテーブル形式で表示
- 走破タイム順での順位表示

---

## 🚀 デプロイメント履歴

### GitHub連携
```bash
# リポジトリ初期化
git init
git add .
git commit -m "Initial commit: Streamlit keiba prediction app"
git branch -M main
git remote add origin https://github.com/Yu10Kumura/keiba-prediction-app.git
git push -u origin main
```

### 主要コミット履歴
1. **27347bf** - Fix infinite loop in CSV processing by adding session state management
2. **dce2420** - Fix tuple error and update deprecated use_container_width parameter  
3. **129058c** - Fix CSV horse name display and prevent infinite loops
4. **2cf6cab** - Force Streamlit Cloud redeploy for latest fixes

### Streamlit Cloud設定
- **Repository:** Yu10Kumura/keiba-prediction-app
- **Branch:** main
- **Main file:** app.py
- **Python version:** 3.11+
- **Dependencies:** requirements.txt

---

## 🔧 技術詳細

### セッション状態管理
```python
# 重要なセッション変数
st.session_state.processed_data      # 処理済みデータ
st.session_state.prediction_results  # 予測結果
st.session_state.original_csv_data   # 元のCSVデータ（馬名保持用）
```

### データフロー
1. **CSVアップロード** → `data_input.py`
2. **血統分類付加** → `bloodline_manager.py`
3. **データ前処理** → `data_processor.py`
4. **予測実行** → `prediction_engine.py`
5. **結果表示** → `result_display.py`

### 予測結果構造
```python
prediction_results = {
    'success': True,
    'predictions': [108.35, 109.12, 107.89, ...],  # 11頭分
    'processing_info': {...}
}
```

---

## 🎯 今後の改善点

### 機能拡張
- [ ] 予測精度の向上（特徴量エンジニアリング）
- [ ] レース結果との比較機能
- [ ] 過去の予測履歴保存
- [ ] ユーザー認証機能

### UI/UX改善
- [ ] レスポンシブデザイン対応
- [ ] 予測結果の可視化（グラフ）
- [ ] エラーハンドリングの強化
- [ ] ローディング表示の改善

### 技術的改善
- [ ] テストコードの追加
- [ ] ログ機能の強化
- [ ] パフォーマンス最適化
- [ ] セキュリティ強化

---

## 📚 学習ポイント

### Streamlit開発のベストプラクティス
1. **セッション状態管理：** `st.rerun()`の使用は最小限に
2. **データ保持：** 元データと処理済みデータを分けて管理
3. **エラーハンドリング：** try-except文での適切な例外処理
4. **デバッグ：** `st.write()`でのデバッグ情報表示

### GitHub/Streamlit Cloud連携
1. **自動デプロイ：** Gitプッシュで自動更新
2. **強制再デプロイ：** ファイル変更でのデプロイトリガー
3. **環境管理：** requirements.txtでの依存関係管理

---

## 💡 トラブルシューティング

### よくある問題
1. **無限ループ：** `st.rerun()`の過度な使用
2. **データ消失：** セッション状態の不適切な管理
3. **表示エラー：** データ型の不一致
4. **デプロイ失敗：** 依存関係の問題

### デバッグ方法
```python
# デバッグ用コード例
st.write(f"DEBUG: データ形状: {data.shape}")
st.write(f"DEBUG: セッション状態: {st.session_state}")
```

---

## 🔗 関連リンク

- **GitHub Repository:** https://github.com/Yu10Kumura/keiba-prediction-app
- **Streamlit Cloud:** https://keiba-prediction-app-sadakai.streamlit.app/
- **ローカル開発:** `streamlit run app.py`

---

## 📝 開発環境

### 必要なツール
- Python 3.11+
- VS Code
- Git
- GitHub Account
- Streamlit Cloud Account

### セットアップ手順
```bash
# 1. リポジトリクローン
git clone https://github.com/Yu10Kumura/keiba-prediction-app.git

# 2. 依存関係インストール
pip install -r requirements.txt

# 3. ローカル実行
streamlit run app.py
```

---

**作成者：** yu10kumura  
**最終更新：** 2025年10月28日  
**バージョン：** v1.0