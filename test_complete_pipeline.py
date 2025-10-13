"""
Integrated test for the complete data processing and prediction pipeline.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.bloodline_manager import BloodlineManager
from src.core.data_processor import DataProcessor, create_sample_data
from src.core.prediction_engine import PredictionEngine, create_prediction_engine
from src.utils.config_manager import get_config


def test_complete_pipeline():
    """Test the complete data processing and prediction pipeline."""
    print("🏇 完全パイプラインテスト開始")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 コンポーネント初期化")
    config = get_config()
    bloodline_manager = BloodlineManager()
    data_processor = DataProcessor()
    prediction_engine = create_prediction_engine()
    
    print(f"✅ 設定管理: 正常")
    print(f"✅ 血統管理: 正常 (マスタデータ: {len(bloodline_manager.bloodline_data)}件)")
    print(f"✅ データ処理: 正常 (エンコーダ: {len(data_processor.label_encoders)}件)")
    print(f"✅ 予測エンジン: {'正常' if prediction_engine.is_model_loaded() else 'モデル未読み込み'}")
    
    # Create test data
    print("\n📊 テストデータ作成")
    sample_df = create_sample_data()
    print(f"サンプルデータ: {sample_df.shape}")
    print("データ内容:")
    print(sample_df.to_string())
    
    # Step 1: Bloodline enrichment
    print("\n🧬 STEP 1: 血統情報付与")
    enriched_df = bloodline_manager.enrich_dataframe(sample_df, '父馬名', '母の父馬名')
    print(f"血統付与後: {enriched_df.shape}")
    new_columns = set(enriched_df.columns) - set(sample_df.columns)
    if new_columns:
        print("追加された血統列:")
        for col in new_columns:
            print(f"  - {col}")
    
    # Step 2: Data processing
    print("\n⚙️ STEP 2: データ前処理")
    processed_df, processing_info = data_processor.process_data(enriched_df)
    print(f"前処理後: {processed_df.shape}")
    print(f"処理時間: {processing_info.get('processing_duration', 0):.3f}秒")
    print(f"バリデーション: {'✅' if processing_info['validation_passed'] else '❌'}")
    print(f"クリーニング: {'✅' if processing_info['cleaning_completed'] else '❌'}")
    print(f"特徴量生成: {'✅' if processing_info['feature_engineering_completed'] else '❌'}")
    
    if processing_info['errors']:
        print("処理エラー:")
        for error in processing_info['errors']:
            print(f"  - {error}")
    
    # Step 3: Feature validation
    print("\n🔍 STEP 3: 特徴量バリデーション")
    is_valid, missing_features = prediction_engine.validate_features(processed_df)
    print(f"特徴量バリデーション: {'✅ 成功' if is_valid else '❌ 失敗'}")
    
    if missing_features:
        print(f"不足特徴量 ({len(missing_features)}個):")
        for feature in missing_features[:10]:  # Show first 10
            print(f"  - {feature}")
        if len(missing_features) > 10:
            print(f"  ... その他 {len(missing_features) - 10} 個")
    
    # Step 4: Prediction (if model available)
    print("\n🔮 STEP 4: 予測実行")
    if prediction_engine.is_model_loaded():
        print("学習済みモデルを使用した予測")
        results_df, batch_info = prediction_engine.predict_batch(processed_df)
        
        print(f"予測結果:")
        print(f"  - 成功: {batch_info['successful_predictions']}件")
        print(f"  - 失敗: {batch_info['failed_predictions']}件")
        print(f"  - 処理時間: {batch_info.get('processing_duration', 0):.3f}秒")
        
        if batch_info['successful_predictions'] > 0:
            pred_col = 'predicted_time_sec'
            if pred_col in results_df.columns:
                predictions = results_df[pred_col].dropna()
                print(f"  - 予測値範囲: {predictions.min():.2f} - {predictions.max():.2f}秒")
                print(f"  - 平均予測値: {predictions.mean():.2f}秒")
    else:
        print("⚠️  モデルが読み込まれていないため予測をスキップ")
        print("   実際の運用時には学習済みモデルが必要です")
    
    # Pipeline summary
    print("\n📋 パイプライン要約")
    print(f"入力データ: {sample_df.shape} → 最終データ: {processed_df.shape}")
    pipeline_success = (
        processing_info['validation_passed'] and
        processing_info['cleaning_completed'] and
        processing_info['feature_engineering_completed']
    )
    print(f"パイプライン成功: {'✅' if pipeline_success else '❌'}")
    
    if pipeline_success:
        print("✅ データ処理パイプラインは正常に動作しています")
        print("   予測機能を有効にするには学習済みモデルが必要です")
    else:
        print("❌ パイプラインにエラーがあります")
    
    print("\n🏇 完全パイプラインテスト完了")
    print("=" * 60)


def test_config_management():
    """Test configuration management."""
    print("\n⚙️ 設定管理テスト")
    config = get_config()
    
    # Test basic config access
    app_title = config.get('app.title', 'Unknown')
    print(f"アプリタイトル: {app_title}")
    
    ui_config = config.get_ui_config()
    print(f"UI設定: {len(ui_config)}項目")
    
    model_config = config.get_model_config()
    print(f"モデル設定: {len(model_config)}項目")
    
    # Test config validation
    is_valid = config.validate_config()
    print(f"設定バリデーション: {'✅ 成功' if is_valid else '❌ 失敗'}")


def show_component_status():
    """Show the status of all components."""
    print("\n🏗️ コンポーネント状況")
    print("-" * 40)
    
    # Configuration
    try:
        config = get_config()
        print("✅ 設定管理: 正常")
    except Exception as e:
        print(f"❌ 設定管理: エラー - {e}")
    
    # Bloodline Manager
    try:
        bloodline_manager = BloodlineManager()
        data_count = len(bloodline_manager.bloodline_data)
        print(f"✅ 血統管理: 正常 ({data_count}件)")
    except Exception as e:
        print(f"❌ 血統管理: エラー - {e}")
    
    # Data Processor
    try:
        data_processor = DataProcessor()
        encoder_count = len(data_processor.label_encoders)
        print(f"⚠️  データ処理: エンコーダ不足 ({encoder_count}件)")
    except Exception as e:
        print(f"❌ データ処理: エラー - {e}")
    
    # Prediction Engine
    try:
        prediction_engine = create_prediction_engine()
        model_status = "正常" if prediction_engine.is_model_loaded() else "モデル未読み込み"
        print(f"⚠️  予測エンジン: {model_status}")
    except Exception as e:
        print(f"❌ 予測エンジン: エラー - {e}")


if __name__ == "__main__":
    show_component_status()
    test_config_management()
    test_complete_pipeline()
