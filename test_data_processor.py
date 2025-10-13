"""
Test file for data processing functionality.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_processor import DataProcessor, create_sample_data
import pandas as pd


def test_data_processor():
    """Test the data processing functionality."""
    print("🧪 データ前処理機能テスト開始")
    
    # Create processor instance
    processor = DataProcessor()
    print(f"✅ DataProcessor初期化完了")
    print(f"   - 読み込み済みエンコーダ数: {len(processor.label_encoders)}")
    print(f"   - 特徴量数: {len(processor.feature_names)}")
    
    # Create sample data
    sample_df = create_sample_data()
    print(f"\n📊 サンプルデータ作成完了: {sample_df.shape}")
    print("サンプルデータ:")
    print(sample_df.head())
    
    # Test validation
    print("\n🔍 データバリデーションテスト")
    is_valid, errors = processor.validate_input_data(sample_df)
    print(f"バリデーション結果: {'✅ 成功' if is_valid else '❌ 失敗'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Test complete processing pipeline
    print("\n⚙️ 完全データ処理パイプラインテスト")
    try:
        processed_data, processing_info = processor.process_data(sample_df)
        
        print("処理結果:")
        print(f"  - 入力レコード数: {processing_info['input_records']}")
        print(f"  - 出力レコード数: {processing_info['output_records']}")
        print(f"  - バリデーション: {'✅' if processing_info['validation_passed'] else '❌'}")
        print(f"  - データクリーニング: {'✅' if processing_info['cleaning_completed'] else '❌'}")
        print(f"  - 特徴量エンジニアリング: {'✅' if processing_info['feature_engineering_completed'] else '❌'}")
        
        if processing_info['errors']:
            print("エラー:")
            for error in processing_info['errors']:
                print(f"  - {error}")
        
        print(f"\n🔧 処理後データ形状: {processed_data.shape}")
        if len(processed_data) > 0:
            print("処理後データ列:")
            for col in processed_data.columns:
                print(f"  - {col}")
    
    except Exception as e:
        print(f"❌ 処理エラー: {e}")
    
    print("\n🧪 データ前処理機能テスト完了")


if __name__ == "__main__":
    test_data_processor()
