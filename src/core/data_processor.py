"""
Data preprocessing functionality for the horse racing simulation app.

This module provides comprehensive data preprocessing capabilities including
data cleaning, feature engineering, validation, and encoding that matches
the training pipeline used for the ML model.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

from ..constants.columns import REQUIRED_COLUMNS, FEATURE_NAMES, CATEGORICAL_FEATURES
from ..constants.validation_rules import VALIDATION_RULES
from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data preprocessing for horse racing prediction.
    
    This class provides comprehensive data preprocessing functionality
    that matches the training pipeline, including data cleaning,
    feature engineering, encoding, and validation.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            model_dir: Path to the directory containing trained models and encoders
        """
        self.config = get_config()
        self.model_dir = model_dir or self._get_default_model_dir()
        self.label_encoders = {}
        self.feature_names = FEATURE_NAMES.copy()
        self._load_encoders()
    
    def _get_default_model_dir(self) -> str:
        """Get the default model directory path."""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "models")
    
    def _load_encoders(self) -> None:
        """Load pre-trained label encoders."""
        try:
            # Get encoder file name from config if available
            try:
                encoder_file = self.config.get('model', {}).get('label_encoders_file', 'label_encoders.pkl')
            except:
                encoder_file = 'label_encoders.pkl'
            
            # Try different encoder file names
            encoder_files = [
                encoder_file,
                "label_encoders.pkl",
                "label_encoders_20251011_225236.pkl"
            ]
            
            encoder_path = None
            for encoder_file_name in encoder_files:
                candidate_path = Path(self.model_dir) / encoder_file_name
                if candidate_path.exists():
                    encoder_path = candidate_path
                    break
            
            if encoder_path is not None:
                with open(encoder_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                logger.info(f"Loaded {len(self.label_encoders)} label encoders from {encoder_path}")
            else:
                logger.warning(f"No encoder file found in {self.model_dir}")
                logger.warning(f"Available files: {list(Path(self.model_dir).glob('*.pkl'))}")
                
        except Exception as e:
            logger.error(f"Failed to load encoders: {e}")
            self.label_encoders = {}
    
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data structure and contents.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Check required columns
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                errors.append(f"必須列が不足しています: {', '.join(missing_cols)}")
            
            # Check data types and ranges
            for col, rules in VALIDATION_RULES.items():
                if col not in df.columns:
                    continue
                
                # Check non-null values if required
                if rules.get('required', False):
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        errors.append(f"{col}: {null_count}件の欠損値があります")
                
                # Check data type
                expected_dtype = rules.get('dtype')
                if expected_dtype and expected_dtype != 'str':
                    # Only check numeric types strictly
                    if expected_dtype == 'int' and not pd.api.types.is_integer_dtype(df[col]):
                        errors.append(f"{col}: データ型が不正です (期待値: {expected_dtype})")
                    elif expected_dtype == 'float' and not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"{col}: データ型が不正です (期待値: {expected_dtype})")
                
                # Check value ranges
                min_val = rules.get('min_value')
                max_val = rules.get('max_value')
                if min_val is not None:
                    invalid_min = (df[col] < min_val).sum()
                    if invalid_min > 0:
                        errors.append(f"{col}: {invalid_min}件が最小値({min_val})未満です")
                
                if max_val is not None:
                    invalid_max = (df[col] > max_val).sum()
                    if invalid_max > 0:
                        errors.append(f"{col}: {invalid_max}件が最大値({max_val})超過です")
                
                # Check allowed values
                allowed_values = rules.get('allowed_values')
                if allowed_values:
                    invalid_values = ~df[col].isin(allowed_values)
                    invalid_count = invalid_values.sum()
                    if invalid_count > 0:
                        errors.append(f"{col}: {invalid_count}件が許可値外です")
            
            # Check date format
            if '年月日' in df.columns:
                try:
                    pd.to_datetime(df['年月日'], format='%Y/%m/%d')
                except:
                    errors.append("年月日: 日付形式が不正です (YYYY/MM/DD形式で入力してください)")
            
            # Data consistency checks
            if len(df) == 0:
                errors.append("データが空です")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, [f"バリデーション処理でエラーが発生しました: {str(e)}"]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Convert date column
            if '年月日' in df_clean.columns:
                df_clean['年月日'] = pd.to_datetime(df_clean['年月日'], format='%Y/%m/%d')
            
            # Remove outliers (same logic as training)
            initial_count = len(df_clean)
            
            # Skip race time processing for prediction data (no target time)
            # (走破タイム処理をスキップ - 予測用データのため)
            zero_count = 0
            outlier_count = 0
            
            logger.info(f"Data cleaning completed:")
            logger.info(f"  Zero time records removed: {zero_count}")
            logger.info(f"  IQR outliers removed: {outlier_count}")
            logger.info(f"  Remaining records: {len(df_clean)} (from {initial_count})")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df_features = df.copy()
            
            # Date features
            if '年月日' in df_features.columns:
                df_features['年'] = df_features['年月日'].dt.year
                df_features['月'] = df_features['年月日'].dt.month
                df_features['日'] = df_features['年月日'].dt.day
                df_features['曜日'] = df_features['年月日'].dt.dayofweek
                df_features['半期'] = df_features['月'].apply(lambda x: 1 if x <= 6 else 2)
            
            # Categorical encoding
            categorical_features = ['場所', '芝・ダ', '馬場状態']
            for col in categorical_features:
                if col in df_features.columns and col in self.label_encoders:
                    try:
                        df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])
                    except ValueError as e:
                        # Handle unseen categories
                        logger.warning(f"Unseen categories in {col}: {e}")
                        # Use most frequent category as fallback
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_features[col] = df_features[col].fillna(most_frequent)
                        unknown_mask = ~df_features[col].isin(self.label_encoders[col].classes_)
                        df_features.loc[unknown_mask, col] = most_frequent
                        df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])
            
            # Bloodline features
            bloodline_features = [
                '父馬名_小系統', '父馬名_国系統', 
                '母の父馬名_小系統', '母の父馬名_国系統'
            ]
            
            for col in bloodline_features:
                if col in df_features.columns:
                    # Fill missing values with 'Unknown'
                    df_features[col] = df_features[col].fillna('Unknown')
                    
                    if col in self.label_encoders:
                        try:
                            df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])
                        except ValueError:
                            # Handle unseen bloodline categories
                            unknown_mask = ~df_features[col].isin(self.label_encoders[col].classes_)
                            df_features.loc[unknown_mask, col] = 'Unknown'
                            df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])
            
            # Bloodline combination features
            if all(col in df_features.columns for col in ['父馬名_小系統', '母の父馬名_小系統']):
                df_features['父母系統組合せ'] = (
                    df_features['父馬名_小系統'] + '_' + 
                    df_features['母の父馬名_小系統']
                )
                
                if '父母系統組合せ' in self.label_encoders:
                    try:
                        df_features['父母系統組合せ_encoded'] = self.label_encoders['父母系統組合せ'].transform(
                            df_features['父母系統組合せ']
                        )
                    except ValueError:
                        # Handle unseen combinations
                        unknown_combo = 'Unknown_Unknown'
                        unknown_mask = ~df_features['父母系統組合せ'].isin(
                            self.label_encoders['父母系統組合せ'].classes_
                        )
                        df_features.loc[unknown_mask, '父母系統組合せ'] = unknown_combo
                        df_features['父母系統組合せ_encoded'] = self.label_encoders['父母系統組合せ'].transform(
                            df_features['父母系統組合せ']
                        )
            
            # Bloodline availability flags
            df_features['父血統有無'] = (~df_features['父馬名_小系統'].eq('Unknown')).astype(int)
            df_features['母父血統有無'] = (~df_features['母の父馬名_小系統'].eq('Unknown')).astype(int)
            
            # Distance category
            def distance_category(distance):
                if distance <= 1200:
                    return 'sprint'
                elif distance <= 1600:
                    return 'mile'
                elif distance <= 2000:
                    return 'classic'
                else:
                    return 'stayer'
            
            if '距離' in df_features.columns:
                df_features['距離カテゴリ'] = df_features['距離'].apply(distance_category)
                
                if '距離カテゴリ' in self.label_encoders:
                    df_features['距離カテゴリ_encoded'] = self.label_encoders['距離カテゴリ'].transform(
                        df_features['距離カテゴリ']
                    )
            
            logger.info("Feature engineering completed successfully")
            return df_features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def prepare_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final input for model prediction.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame ready for model input
        """
        try:
            # Select only the features used by the model
            available_features = [col for col in self.feature_names if col in df.columns]
            
            if len(available_features) != len(self.feature_names):
                missing_features = set(self.feature_names) - set(available_features)
                logger.warning(f"Missing features: {missing_features}")
            
            model_input = df[available_features].copy()
            
            # Handle any remaining missing values
            model_input = model_input.fillna(0)
            
            logger.info(f"Model input prepared: {model_input.shape}")
            return model_input
            
        except Exception as e:
            logger.error(f"Model input preparation failed: {e}")
            raise
    
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete data processing pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Tuple of (processed_data, processing_info)
        """
        try:
            processing_info = {
                'start_time': datetime.now(),
                'input_records': len(df),
                'validation_passed': False,
                'cleaning_completed': False,
                'feature_engineering_completed': False,
                'output_records': 0,
                'errors': []
            }
            
            # Step 1: Validation
            is_valid, validation_errors = self.validate_input_data(df)
            processing_info['validation_passed'] = is_valid
            processing_info['errors'].extend(validation_errors)
            
            if not is_valid:
                logger.error("Data validation failed")
                return df, processing_info
            
            # Step 2: Data cleaning
            df_clean = self.clean_data(df)
            processing_info['cleaning_completed'] = True
            
            # Step 3: Feature engineering
            df_features = self.engineer_features(df_clean)
            processing_info['feature_engineering_completed'] = True
            
            # Step 4: Model input preparation
            df_final = self.prepare_model_input(df_features)
            
            processing_info['output_records'] = len(df_final)
            processing_info['end_time'] = datetime.now()
            processing_info['processing_duration'] = (
                processing_info['end_time'] - processing_info['start_time']
            ).total_seconds()
            
            logger.info("Data processing pipeline completed successfully")
            return df_final, processing_info
            
        except Exception as e:
            logger.error(f"Data processing pipeline failed: {e}")
            processing_info['errors'].append(f"処理エラー: {str(e)}")
            return df, processing_info


def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing purposes.
    
    Returns:
        Sample DataFrame with proper structure
    """
    sample_data = pd.DataFrame({
        '年月日': ['2024/12/25', '2024/12/26'],
        '場所': ['東京', '中山'],
        '距離': [1600, 2000],
        '芝・ダ': ['芝', 'ダ'],
        '馬場状態': ['良', '稍重'],
        '馬番': [1, 2],
        '走破タイム': [95.5, 120.8],
        '父馬名_小系統': ['サンデーサイレンス系', 'ノーザンダンサー系'],
        '父馬名_国系統': ['アメリカ系', 'カナダ系'],
        '母の父馬名_小系統': ['ミスタープロスペクター系', 'サンデーサイレンス系'],
        '母の父馬名_国系統': ['アメリカ系', 'アメリカ系']
    })
    
    return sample_data
