"""
Check the actual feature names used during training.
"""

import pickle
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_model_features():
    """Check the features used in the trained model."""
    print("🔍 学習済みモデルの特徴量チェック")
    print("=" * 50)
    
    # Load feature list from trained model
    feature_list_path = "models/feature_list.pkl"
    try:
        with open(feature_list_path, 'rb') as f:
            trained_features = pickle.load(f)
        
        print(f"✅ 学習時の特徴量リスト読み込み成功")
        print(f"   特徴量数: {len(trained_features)}")
        print("\n学習時の特徴量:")
        for i, feature in enumerate(trained_features, 1):
            print(f"{i:2d}. {feature}")
        
        return trained_features
        
    except Exception as e:
        print(f"❌ 特徴量リスト読み込み失敗: {e}")
        return None

def check_label_encoders():
    """Check the label encoders."""
    print("\n🏷️ ラベルエンコーダチェック")
    print("=" * 50)
    
    encoder_path = "models/label_encoders.pkl"
    try:
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        
        print(f"✅ ラベルエンコーダ読み込み成功")
        print(f"   エンコーダ数: {len(encoders)}")
        print("\nエンコーダ一覧:")
        for i, (key, encoder) in enumerate(encoders.items(), 1):
            classes = getattr(encoder, 'classes_', None)
            class_count = len(classes) if classes is not None else 'Unknown'
            print(f"{i:2d}. {key}: {class_count}種類")
        
        return encoders
        
    except Exception as e:
        print(f"❌ ラベルエンコーダ読み込み失敗: {e}")
        return None

def check_model():
    """Check the trained model."""
    print("\n🤖 学習済みモデルチェック")
    print("=" * 50)
    
    model_path = "models/lgb_model.pkl"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ モデル読み込み成功")
        print(f"   モデル型: {type(model).__name__}")
        
        # Try to get model info
        if hasattr(model, 'num_feature'):
            print(f"   特徴量数: {model.num_feature()}")
        if hasattr(model, 'feature_name'):
            feature_names = model.feature_name()
            print(f"   モデル内特徴量数: {len(feature_names)}")
            
        return model
        
    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return None

def compare_with_current_constants():
    """Compare with current feature constants."""
    print("\n🔄 現在の定数との比較")
    print("=" * 50)
    
    # Current feature names from constants
    current_features = [
        '距離', '馬番', '年', '月', '曜日', '半期',
        '場所_encoded', '芝・ダ_encoded', '馬場状態_encoded',
        '父馬名_小系統_encoded', '父馬名_国系統_encoded',
        '母の父馬名_小系統_encoded', '母の父馬名_国系統_encoded',
        '父母系統組合せ_encoded', '距離カテゴリ_encoded',
        '父血統有無', '母父血統有無'
    ]
    
    # Load trained features
    try:
        with open("models/feature_list.pkl", 'rb') as f:
            trained_features = pickle.load(f)
        
        print("比較結果:")
        print(f"  現在定義: {len(current_features)}個")
        print(f"  学習時: {len(trained_features)}個")
        
        # Check differences
        current_set = set(current_features)
        trained_set = set(trained_features)
        
        missing_in_current = trained_set - current_set
        extra_in_current = current_set - trained_set
        
        if missing_in_current:
            print(f"\n⚠️  現在定義に不足している特徴量:")
            for feature in missing_in_current:
                print(f"    - {feature}")
        
        if extra_in_current:
            print(f"\n⚠️  現在定義の余分な特徴量:")
            for feature in extra_in_current:
                print(f"    - {feature}")
        
        if not missing_in_current and not extra_in_current:
            print("✅ 完全一致！")
        
        return trained_features, current_features
        
    except Exception as e:
        print(f"❌ 比較失敗: {e}")
        return None, current_features

if __name__ == "__main__":
    trained_features = check_model_features()
    encoders = check_label_encoders()
    model = check_model()
    trained_features, current_features = compare_with_current_constants()
    
    print("\n📋 総合判定")
    print("=" * 50)
    
    all_good = True
    if trained_features is None:
        print("❌ 特徴量リストが読み込めません")
        all_good = False
    if encoders is None:
        print("❌ ラベルエンコーダが読み込めません")
        all_good = False
    if model is None:
        print("❌ モデルが読み込めません")
        all_good = False
    
    if all_good:
        print("✅ 全てのモデルファイルが正常に読み込めます")
        print("🚀 予測エンジンのテストが実行可能です")
