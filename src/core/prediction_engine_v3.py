"""
V3 Phase 1統合版対応の予測エンジン

V3の新機能:
- Phase 1改善版脚質推定
- レース展開・騎手統計特徴量
- 順位予測機能
"""

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

from ..constants.columns import V3_FEATURE_NAMES, PREDICTION_COLUMN
from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)


class PredictionEngineV3:
    """
    V3 Phase 1統合版予測エンジン
    
    新機能:
    - 改善された脚質推定
    - 順位予測機能
    - レース展開分析
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        V3予測エンジンの初期化
        
        Args:
            model_dir: モデルディレクトリパス
        """
        self.config = get_config()
        self.model_dir = model_dir or self._get_v3_model_dir()
        self.model = None
        self.model_info = {}
        self.feature_names = V3_FEATURE_NAMES
        self._load_v3_model()
    
    def _get_v3_model_dir(self) -> str:
        """V3モデルディレクトリのパス取得"""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        return str(project_root / "ml_models_v3" / "models")
    
    def _load_v3_model(self) -> None:
        """V3モデル読み込み"""
        try:
            model_dir_path = Path(self.model_dir)
            
            # V3モデルファイル検索
            v3_model_files = list(model_dir_path.glob("keiba_model_v3_*.pkl"))
            
            if not v3_model_files:
                logger.error(f"V3モデルファイルが見つかりません: {model_dir_path}")
                return
            
            # 最新のV3モデル取得
            latest_model_path = max(v3_model_files, key=lambda p: p.stat().st_ctime)
            
            # モデル読み込み
            with open(latest_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 辞書形式の場合、モデル本体を抽出
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.model_metadata = model_data
            else:
                self.model = model_data
                self.model_metadata = {}
            
            self.model_info = {
                'model_type': 'LightGBM V3 Phase 1',
                'version': 'V3_Phase1',
                'loaded_at': datetime.now(),
                'model_path': str(latest_model_path),
                'feature_count': len(self.feature_names)
            }
            
            logger.info(f"V3モデル読み込み成功: {latest_model_path.name}")
            logger.info(f"特徴量数: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"V3モデル読み込み失敗: {e}")
            self.model = None
    
    def estimate_running_style_improved(self, row: pd.Series) -> int:
        """
        Phase 1改善版脚質推定
        
        Args:
            row: レースデータ行
            
        Returns:
            脚質コード (0:逃げ, 1:先行, 2:差し, 3:追込)
        """
        try:
            # 通過順データ取得
            positions = []
            for i in range(1, 5):
                col = f'通過順{i}'
                if col in row and pd.notna(row[col]) and row[col] > 0:
                    positions.append(row[col])
                    
            if len(positions) < 3:
                return 1  # デフォルト：先行
                
            # 頭数で正規化
            head_count = row.get('頭数', 16)
            if pd.isna(head_count) or head_count <= 0:
                head_count = 16
            norm_positions = [pos / head_count for pos in positions]
            
            # 指標計算
            early_pos = norm_positions[0]
            mid_pos = np.mean(norm_positions[1:3]) if len(norm_positions) >= 3 else norm_positions[1]
            late_pos = norm_positions[-1]
            
            late_change = late_pos - mid_pos
            late_kick = -late_change if late_change < 0 else 0
            
            # 改善されたルールベース分類
            if early_pos <= 0.25:  # 前1/4以内
                if late_change <= 0.1:
                    return 0  # 逃げ
                else:
                    return 1  # 先行
            else:  # 前1/4以下
                if late_kick > 0.15:
                    return 3  # 追込
                elif late_kick > 0.05:
                    return 2  # 差し
                elif early_pos <= 0.6:
                    return 1  # 先行
                else:
                    return 2  # 差し
                    
        except Exception as e:
            logger.warning(f"脚質推定エラー: {e}")
            return 1  # エラー時デフォルト
    
    def prepare_v3_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V3用特徴量準備
        
        Args:
            df: 入力データ
            
        Returns:
            V3特徴量準備済みデータ
        """
        try:
            data = df.copy()
            
            # 基本特徴量
            data['距離_log'] = np.log1p(data['距離'])
            data['馬番_norm'] = data['馬番'] / data['頭数']
            data['斤量_norm'] = (data['斤量'] - 50) / 10
            
            # Phase 1改善版脚質推定
            if all(col in data.columns for col in ['通過順1', '通過順2', '通過順3', '通過順4', '頭数']):
                data['脚質'] = data.apply(self.estimate_running_style_improved, axis=1)
            else:
                # 通過順データがない場合の簡易推定
                data['脚質'] = 1  # デフォルト：先行
            
            # カテゴリカル変数エンコーディング
            categorical_cols = ['場所', '芝・ダ', '馬場状態', '父馬名_小系統', '母の父馬名_小系統', 
                               '父馬名_国系統', '母の父馬名_国系統', 'クラスコード']
            
            for col in categorical_cols:
                if col in data.columns:
                    # 簡易エンコーディング（実際のアプリケーションでは既存のエンコーダーを使用）
                    unique_values = data[col].fillna('Unknown').unique()
                    encoding_map = {val: i for i, val in enumerate(unique_values)}
                    data[col + '_encoded'] = data[col].fillna('Unknown').map(encoding_map)
                else:
                    data[col + '_encoded'] = 0
            
            # 騎手・調教師統計（簡易版 - 実際のアプリでは過去データから計算）
            data['騎手勝率'] = 0.074  # 平均値
            data['調教師勝率'] = 0.074
            data['騎手平均タイム'] = data.get('走破タイム', 80.0)  # 簡易版
            data['調教師平均タイム'] = data.get('走破タイム', 80.0)
            
            # 上がり3F標準化
            if '上がり3Fタイム' in data.columns:
                data['上がり3F_標準化'] = (data['上がり3Fタイム'] - data['上がり3Fタイム'].mean()) / data['上がり3Fタイム'].std()
                data['上がり3F_標準化'] = data['上がり3F_標準化'].fillna(0)
            else:
                data['上がり3F_標準化'] = 0
            
            # ペース指標（簡易版）
            data['ペース指標'] = 5.0  # 中庸値
            
            # V3特有の交互作用項
            data['距離_surface'] = data['距離_log'] * data['芝・ダ_encoded']
            data['距離_condition'] = data['距離_log'] * data['馬場状態_encoded']
            data['距離_venue'] = data['距離_log'] * data['場所_encoded']
            data['脚質_距離'] = data['脚質'] * data['距離_log']
            data['脚質_馬場'] = data['脚質'] * data['馬場状態_encoded']
            data['騎手勝率_脚質'] = data['騎手勝率'] * data['脚質']
            data['上がり3F_脚質'] = data['上がり3F_標準化'] * data['脚質']
            data['ペース_脚質'] = data['ペース指標'] * data['脚質']
            
            # 不足特徴量を0で補完
            for feature in self.feature_names:
                if feature not in data.columns:
                    data[feature] = 0
            
            # 特徴量順序を正しくソート
            feature_data = data[self.feature_names]
            
            return feature_data
            
        except Exception as e:
            logger.error(f"V3特徴量準備エラー: {e}")
            raise
    
    def predict_race_order(self, race_data: pd.DataFrame) -> Dict[str, Any]:
        """
        レース内順位予測
        
        Args:
            race_data: 同一レース内の馬データ
            
        Returns:
            順位予測結果
        """
        try:
            if not self.is_model_loaded():
                return {
                    'success': False,
                    'error': 'V3モデルが読み込まれていません'
                }
            
            # V3特徴量準備
            feature_data = self.prepare_v3_features(race_data)
            
            # タイム予測
            predicted_times = self.model.predict(feature_data.values)
            
            # 順位算出（タイムの昇順）
            predicted_order = np.argsort(predicted_times) + 1
            
            # 結果整理
            results = []
            for i, (idx, row) in enumerate(race_data.iterrows()):
                horse_result = {
                    'horse_name': row.get('馬名', f'馬{i+1}'),
                    'horse_number': row.get('馬番', i+1),
                    'predicted_time': float(predicted_times[i]),
                    'predicted_order': int(predicted_order[i]),
                    'running_style': int(feature_data.iloc[i]['脚質']),
                    'running_style_name': self._get_running_style_name(int(feature_data.iloc[i]['脚質']))
                }
                results.append(horse_result)
            
            # 順位順でソート
            results.sort(key=lambda x: x['predicted_order'])
            
            return {
                'success': True,
                'race_results': results,
                'race_summary': {
                    'total_horses': len(race_data),
                    'prediction_time': datetime.now(),
                    'model_version': 'V3_Phase1'
                }
            }
            
        except Exception as e:
            logger.error(f"レース順位予測エラー: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_running_style_name(self, style_code: int) -> str:
        """脚質コードから名前を取得"""
        style_map = {0: '逃げ', 1: '先行', 2: '差し', 3: '追込'}
        return style_map.get(style_code, '不明')
    
    def predict_single(self, feature_data: Union[pd.Series, Dict]) -> Dict[str, Any]:
        """
        単一予測（V3版）
        
        Args:
            feature_data: 特徴量データ
            
        Returns:
            予測結果
        """
        try:
            if not self.is_model_loaded():
                return {
                    'success': False,
                    'error': 'V3モデルが読み込まれていません'
                }
            
            # DataFrameに変換
            if isinstance(feature_data, dict):
                feature_data = pd.Series(feature_data)
            
            single_df = pd.DataFrame([feature_data])
            
            # V3特徴量準備
            prepared_features = self.prepare_v3_features(single_df)
            
            # 予測実行
            prediction = self.model.predict(prepared_features.values)[0]
            
            # 脚質情報取得
            running_style = int(prepared_features.iloc[0]['脚質'])
            
            return {
                'success': True,
                'prediction': float(prediction),
                'predictions': [float(prediction)],
                'confidence': self._calculate_v3_confidence(prediction, feature_data),
                'running_style': running_style,
                'running_style_name': self._get_running_style_name(running_style),
                'model_version': 'V3_Phase1',
                'prediction_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"V3単一予測エラー: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_v3_confidence(self, prediction: float, feature_data: pd.Series) -> Dict[str, Any]:
        """
        V3版信頼度計算
        
        Args:
            prediction: 予測値
            feature_data: 特徴量データ
            
        Returns:
            信頼度情報
        """
        try:
            # 基本信頼度
            if 50 <= prediction <= 250:
                base_confidence = 0.8
                factors = ['予測タイムが妥当範囲内']
            else:
                base_confidence = 0.4
                factors = ['予測タイムが範囲外']
            
            # 特徴量完全性
            available_features = sum(1 for col in ['距離', '馬番', '頭数', '場所', '芝・ダ'] 
                                   if col in feature_data.index and pd.notna(feature_data[col]))
            feature_completeness = available_features / 5 * 0.2
            
            if available_features >= 4:
                factors.append('基本情報が充実')
            else:
                factors.append('基本情報が不足')
            
            # V3特有の信頼度要素
            v3_bonus = 0.0
            if '通過順1' in feature_data.index and pd.notna(feature_data['通過順1']):
                v3_bonus += 0.1
                factors.append('脚質推定データあり')
            
            total_confidence = min(1.0, base_confidence + feature_completeness + v3_bonus)
            
            return {
                'confidence_score': total_confidence,
                'confidence_level': 'high' if total_confidence >= 0.8 else 'medium' if total_confidence >= 0.6 else 'low',
                'factors': factors,
                'v3_features_used': True
            }
            
        except Exception as e:
            logger.warning(f"V3信頼度計算エラー: {e}")
            return {
                'confidence_score': 0.5,
                'confidence_level': 'medium',
                'factors': ['信頼度計算エラー'],
                'v3_features_used': False
            }
    
    def predict(self, data: Union[pd.DataFrame, pd.Series, Dict]) -> Dict[str, Any]:
        """
        メイン予測メソッド（V3版）
        
        Args:
            data: 入力データ
            
        Returns:
            予測結果
        """
        try:
            if isinstance(data, pd.DataFrame):
                if len(data) == 1:
                    # 単一行
                    return self.predict_single(data.iloc[0])
                else:
                    # 複数行（レース予測として処理）
                    return self.predict_race_order(data)
            else:
                # 単一予測
                return self.predict_single(data)
                
        except Exception as e:
            logger.error(f"V3予測エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_version': 'V3_Phase1'
            }
    
    def is_model_loaded(self) -> bool:
        """モデル読み込み状態確認"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        if not self.is_model_loaded():
            return {'status': 'not_loaded', 'version': 'V3_Phase1'}
        
        return {
            'status': 'loaded',
            'version': 'V3_Phase1',
            'model_type': self.model_info.get('model_type', 'LightGBM V3'),
            'loaded_at': self.model_info.get('loaded_at'),
            'feature_count': len(self.feature_names),
            'new_features': ['脚質', '騎手勝率', '調教師勝率', '上がり3F_標準化', 'ペース指標'],
            'improvements': ['Phase 1改善版脚質推定', '順位予測機能', 'レース展開分析']
        }


def create_prediction_engine_v3() -> PredictionEngineV3:
    """
    V3予測エンジンインスタンス作成
    
    Returns:
        設定済みV3予測エンジン
    """
    return PredictionEngineV3()
