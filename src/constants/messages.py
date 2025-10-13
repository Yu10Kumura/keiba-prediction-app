# メッセージ定義
"""
アプリケーションで使用するメッセージの定数定義
"""

# 成功メッセージ
SUCCESS_MESSAGES = {
    'file_uploaded': 'ファイルが正常にアップロードされました',
    'prediction_completed': '予測が完了しました',
    'bloodline_found': '血統情報が見つかりました',
    'data_saved': 'データが保存されました'
}

# エラーメッセージ
ERROR_MESSAGES = {
    # ファイル関連
    'file_not_found': 'ファイルが見つかりません',
    'file_too_large': 'ファイルサイズが上限（50MB）を超えています',
    'file_format_error': 'サポートされていないファイル形式です（CSV, Excelのみ対応）',
    'file_encoding_error': 'ファイルの文字エンコーディングが読み取れません',
    'file_read_error': 'ファイルの読み込みに失敗しました',
    
    # データ検証関連
    'missing_required_columns': '必須列が不足しています: {}',
    'invalid_data_type': 'データ型が正しくありません: {}',
    'value_out_of_range': '値が範囲外です: {}',
    'invalid_venue': '無効な競馬場名です: {}',
    'invalid_surface': '芝・ダは「芝」または「ダ」を入力してください',
    'invalid_track_condition': '馬場状態は「良」「稍重」「重」「不良」のいずれかを入力してください',
    'too_many_rows': '行数が上限（{}行）を超えています',
    
    # 日付関連
    'invalid_date_format': '日付形式が正しくありません（YYYY/MM/DD形式で入力してください）',
    'date_out_of_range': '日付が範囲外です（1990年〜2030年）',
    
    # 予測関連
    'model_load_error': 'モデルの読み込みに失敗しました',
    'prediction_error': '予測処理中にエラーが発生しました',
    'bloodline_lookup_error': '血統情報の検索に失敗しました',
    
    # システム関連
    'memory_error': 'メモリ不足により処理を継続できません',
    'unknown_error': '予期しないエラーが発生しました'
}

# 警告メッセージ
WARNING_MESSAGES = {
    'bloodline_not_found': '血統情報が見つかりません（{}）。「UNK」で代替します',
    'partial_bloodline_found': '一部の血統情報が見つかりません',
    'large_file_processing': '大きなファイルのため処理に時間がかかる場合があります',
    'encoding_auto_detected': 'ファイルエンコーディングを自動判定しました: {}'
}

# 情報メッセージ
INFO_MESSAGES = {
    'processing_start': '処理を開始します...',
    'processing_complete': '処理が完了しました',
    'bloodline_enrichment': '血統情報を付与しています...',
    'model_prediction': 'モデルで予測中...',
    'file_preview': 'ファイルプレビュー（先頭10行）',
    'prediction_summary': '予測結果サマリー'
}

# UI関連メッセージ
UI_MESSAGES = {
    'app_title': '🏇 競馬走破タイム予測アプリ',
    'tab_file_upload': '📁 ファイルアップロード',
    'tab_manual_input': '✏️ 手入力',
    'sidebar_title': '⚙️ 設定',
    'history_title': '📜 予測履歴',
    'download_button': '📥 結果をダウンロード',
    'predict_button': '🔮 予測実行',
    'add_to_history': '📋 履歴に追加',
    'clear_history': '🗑️ 履歴をクリア'
}
