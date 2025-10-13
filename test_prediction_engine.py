"""
Test file for prediction engine functionality.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.prediction_engine import PredictionEngine, create_prediction_engine
from src.core.data_processor import DataProcessor, create_sample_data


def test_prediction_engine():
    """Test the prediction engine functionality."""
    print("🔮 予測エンジンテスト開始")
    
    # Create prediction engine
    engine = create_prediction_engine()
    print(f"✅ PredictionEngine初期化完了")
    print(f"   - モデル読み込み状況: {'✅ 成功' if engine.is_model_loaded() else '❌ 失敗'}")
    
    # Get model info
    model_info = engine.get_model_info()
    print(f"   - モデル状況: {model_info['status']}")
    if model_info['status'] == 'loaded':
        print(f"   - モデル種類: {model_info['model_type']}")
        print(f"   - 特徴量数: {model_info['feature_count']}")
    
    if not engine.is_model_loaded():
        print("⚠️  モデルが読み込まれていません。テストを続行しますが、予測は失敗します。")
    
    # Create sample data and process it
    print("\n📊 サンプルデータ準備")
    processor = DataProcessor()
    sample_df = create_sample_data()
    
    # Process the data to get proper features
    processed_data, processing_info = processor.process_data(sample_df)
    print(f"   - 処理前: {sample_df.shape}")
    print(f"   - 処理後: {processed_data.shape}")
    print(f"   - 処理成功: {'✅' if processing_info['feature_engineering_completed'] else '❌'}")
    
    if len(processed_data) == 0:
        print("❌ 処理後データが空です。テストを終了します。")
        return
    
    # Test feature validation
    print("\n🔍 特徴量バリデーションテスト")
    is_valid, missing_features = engine.validate_features(processed_data)
    print(f"バリデーション結果: {'✅ 成功' if is_valid else '❌ 失敗'}")
    if missing_features:
        print("不足特徴量:")
        for feature in missing_features[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(missing_features) > 5:
            print(f"  ... その他 {len(missing_features) - 5} 個")
    
    # Test single prediction
    print("\n🎯 単一予測テスト")
    if len(processed_data) > 0:
        first_row = processed_data.iloc[0]
        prediction_result = engine.predict_single(first_row)
        
        print(f"予測結果: {'✅ 成功' if prediction_result['success'] else '❌ 失敗'}")
        if prediction_result['success']:
            print(f"  - 予測値: {prediction_result['prediction']:.2f}秒")
            confidence = prediction_result['confidence']
            print(f"  - 信頼度: {confidence['confidence_score']:.2f} ({confidence['confidence_level']})")
            print(f"  - 考慮要因数: {len(confidence['factors'])}")
        else:
            print(f"  - エラー: {prediction_result['error']}")
    
    # Test batch prediction
    print("\n📦 バッチ予測テスト")
    if len(processed_data) > 0:
        results_df, batch_info = engine.predict_batch(processed_data)
        
        print(f"バッチ処理結果:")
        print(f"  - 入力レコード数: {batch_info['input_records']}")
        print(f"  - 成功予測数: {batch_info['successful_predictions']}")
        print(f"  - 失敗予測数: {batch_info['failed_predictions']}")
        print(f"  - 処理時間: {batch_info.get('processing_duration', 0):.3f}秒")
        
        if batch_info['errors']:
            print("エラー:")
            for error in batch_info['errors']:
                print(f"  - {error}")
        
        if batch_info['successful_predictions'] > 0:
            print(f"\n予測結果データ形状: {results_df.shape}")
            print("新しく追加された列:")
            new_columns = set(results_df.columns) - set(processed_data.columns)
            for col in new_columns:
                print(f"  - {col}")
    
    # Test prediction with explanation
    print("\n📝 詳細説明付き予測テスト")
    if len(processed_data) > 0:
        first_row = processed_data.iloc[0]
        explained_result = engine.predict_with_explanation(first_row)
        
        print(f"説明付き予測: {'✅ 成功' if explained_result['success'] else '❌ 失敗'}")
        if explained_result['success'] and explained_result.get('explanation'):
            explanation = explained_result['explanation']
            print(f"  - 主要要因数: {len(explanation['key_factors'])}")
            print("  - 主要要因:")
            for factor in explanation['key_factors']:
                print(f"    * {factor}")
            print(f"  - 特徴量影響数: {len(explanation['feature_impacts'])}")
    
    print("\n🔮 予測エンジンテスト完了")


def create_mock_processed_data():
    """Create mock processed data for testing when models are not available."""
    mock_data = pd.DataFrame({
        '距離': [1600, 2000],
        '馬番': [1, 2],
        '年': [2024, 2024],
        '月': [12, 12],
        '曜日': [1, 2],
        '半期': [2, 2],
        '場所_encoded': [0, 1],
        '芝・ダ_encoded': [0, 1],
        '馬場状態_encoded': [0, 1],
        '父馬名_小系統_encoded': [0, 1],
        '父馬名_国系統_encoded': [0, 1],
        '母の父馬名_小系統_encoded': [0, 1],
        '母の父馬名_国系統_encoded': [0, 1],
        '父母系統組合せ_encoded': [0, 1],
        '距離カテゴリ_encoded': [0, 1],
        '父血統有無': [1, 1],
        '母父血統有無': [1, 1]
    })
    return mock_data


def test_with_mock_data():
    """Test prediction engine with mock data."""
    print("\n🧪 モックデータでの予測エンジンテスト")
    
    engine = create_prediction_engine()
    mock_data = create_mock_processed_data()
    
    print(f"モックデータ形状: {mock_data.shape}")
    print(f"モデル読み込み状況: {'✅' if engine.is_model_loaded() else '❌'}")
    
    # Test feature validation
    is_valid, missing_features = engine.validate_features(mock_data)
    print(f"特徴量バリデーション: {'✅ 成功' if is_valid else '❌ 失敗'}")
    
    if not is_valid:
        print("不足特徴量:")
        for feature in missing_features:
            print(f"  - {feature}")
    
    # Test prediction (will fail if model not loaded, but tests the flow)
    if engine.is_model_loaded():
        results_df, batch_info = engine.predict_batch(mock_data)
        print(f"予測成功数: {batch_info['successful_predictions']}")
    else:
        print("モデル未読み込みのため予測テストをスキップ")


if __name__ == "__main__":
    test_prediction_engine()
    test_with_mock_data()
