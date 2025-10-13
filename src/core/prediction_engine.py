"""
Prediction engine for the horse racing simulation app.

This module provides prediction functionality using trained LightGBM models
with proper error handling, confidence scoring, and batch processing capabilities.
"""

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from ..constants.columns import FEATURE_NAMES, PREDICTION_COLUMN
from ..utils.config_manager import get_config

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Handles prediction using trained machine learning models.
    
    This class provides prediction functionality with confidence scoring,
    batch processing, and proper error handling for horse racing time prediction.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the prediction engine.
        
        Args:
            model_dir: Path to the directory containing trained models
        """
        self.config = get_config()
        self.model_dir = model_dir or self._get_default_model_dir()
        self.model = None
        self.model_info = {}
        self.feature_names = FEATURE_NAMES.copy()
        self._load_model()
    
    def _get_default_model_dir(self) -> str:
        """Get the default model directory path."""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "models")
    
    def _load_model(self) -> None:
        """Load the trained LightGBM model and metadata."""
        try:
            # Get model file name from config if available
            try:
                model_file = self.config.get('model', {}).get('lgb_model_file', 'lgb_model.pkl')
            except:
                model_file = 'lgb_model.pkl'
            
            # Try different model file names
            model_files = [
                model_file,
                "lgb_model.pkl",
                "keiba_timeseries_lgb_20251011_225236.pkl"
            ]
            
            model_path = None
            for model_file_name in model_files:
                candidate_path = Path(self.model_dir) / model_file_name
                if candidate_path.exists():
                    model_path = candidate_path
                    break
            
            if model_path is None:
                logger.error(f"No model file found in {self.model_dir}")
                logger.error(f"Available files: {list(Path(self.model_dir).glob('*.pkl'))}")
                return
            
            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load model metadata if available
            info_path = Path(self.model_dir) / "lgb_model.txt"
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Parse model info (simplified)
                    self.model_info = {
                        'model_type': 'LightGBM',
                        'loaded_at': datetime.now(),
                        'model_path': str(model_path),
                        'info_content': content
                    }
            else:
                self.model_info = {
                    'model_type': 'LightGBM',
                    'loaded_at': datetime.now(),
                    'model_path': str(model_path)
                }
            
            logger.info(f"Model loaded successfully: {model_path}")
            logger.info(f"Model type: {type(self.model)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def is_model_loaded(self) -> bool:
        """Check if the model is properly loaded."""
        return self.model is not None
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the input DataFrame has required features.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        try:
            missing_features = []
            
            # Check for required feature columns
            for feature in self.feature_names:
                if feature not in df.columns:
                    missing_features.append(feature)
            
            # Check for non-empty data
            if len(df) == 0:
                missing_features.append("Empty DataFrame")
            
            # Check for data types
            for feature in self.feature_names:
                if feature in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[feature]):
                        missing_features.append(f"{feature} (non-numeric)")
            
            return len(missing_features) == 0, missing_features
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def calculate_prediction_confidence(self, prediction: float, feature_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate confidence metrics for a prediction.
        
        Args:
            prediction: Predicted value
            feature_data: Feature values used for prediction
            
        Returns:
            Dictionary containing confidence metrics
        """
        try:
            confidence_info = {
                'confidence_score': 0.0,
                'confidence_level': 'low',
                'factors': []
            }
            
            # Basic confidence based on prediction range
            if 50 <= prediction <= 250:  # Reasonable time range
                base_confidence = 0.7
                confidence_info['factors'].append('Prediction in reasonable range')
            else:
                base_confidence = 0.3
                confidence_info['factors'].append('Prediction outside typical range')
            
            # Distance-based confidence
            distance = feature_data.get('距離', 0)
            if 1000 <= distance <= 3000:  # Common race distances
                distance_confidence = 0.2
                confidence_info['factors'].append('Standard race distance')
            else:
                distance_confidence = 0.1
                confidence_info['factors'].append('Unusual race distance')
            
            # Feature completeness confidence
            non_zero_features = (feature_data != 0).sum()
            feature_completeness = min(1.0, non_zero_features / len(self.feature_names))
            feature_confidence = feature_completeness * 0.1
            
            if feature_completeness > 0.8:
                confidence_info['factors'].append('High feature completeness')
            elif feature_completeness > 0.5:
                confidence_info['factors'].append('Moderate feature completeness')
            else:
                confidence_info['factors'].append('Low feature completeness')
            
            # Calculate total confidence
            total_confidence = base_confidence + distance_confidence + feature_confidence
            confidence_info['confidence_score'] = min(1.0, total_confidence)
            
            # Determine confidence level
            if confidence_info['confidence_score'] >= 0.8:
                confidence_info['confidence_level'] = 'high'
            elif confidence_info['confidence_score'] >= 0.6:
                confidence_info['confidence_level'] = 'medium'
            else:
                confidence_info['confidence_level'] = 'low'
            
            return confidence_info
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return {
                'confidence_score': 0.0,
                'confidence_level': 'unknown',
                'factors': ['Error in confidence calculation']
            }
    
    def predict_single(self, feature_data: Union[pd.Series, Dict]) -> Dict[str, Any]:
        """
        Make prediction for a single race.
        
        Args:
            feature_data: Feature values for prediction
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if not self.is_model_loaded():
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'prediction': None,
                    'confidence': None
                }
            
            # Convert to Series if needed
            if isinstance(feature_data, dict):
                feature_data = pd.Series(feature_data)
            
            # Prepare feature vector with proper column names
            feature_vector = []
            for feature in self.feature_names:
                value = feature_data.get(feature, 0)
                feature_vector.append(value)
            
            # Create DataFrame with proper column names for prediction
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
            
            # Make prediction using DataFrame with column names
            prediction = self.model.predict(feature_df)[0]
            
            # Calculate confidence
            confidence_info = self.calculate_prediction_confidence(prediction, feature_data)
            
            return {
                'success': True,
                'prediction': float(prediction),
                'predictions': [float(prediction)],  # Add this for compatibility
                'confidence': confidence_info,
                'feature_vector': feature_vector,
                'prediction_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }
    
    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Make predictions for multiple races.
        
        Args:
            df: DataFrame with feature data for multiple races
            
        Returns:
            Tuple of (results_dataframe, batch_info)
        """
        try:
            batch_info = {
                'start_time': datetime.now(),
                'input_records': len(df),
                'successful_predictions': 0,
                'failed_predictions': 0,
                'errors': []
            }
            
            if not self.is_model_loaded():
                batch_info['errors'].append('Model not loaded')
                return df, batch_info
            
            # Validate features
            is_valid, missing_features = self.validate_features(df)
            if not is_valid:
                error_msg = f"Missing features: {missing_features}"
                batch_info['errors'].append(error_msg)
                logger.error(error_msg)
                return df, batch_info
            
            # Prepare results DataFrame
            results_df = df.copy()
            predictions = []
            confidence_scores = []
            confidence_levels = []
            
            # Make predictions
            for idx, row in df.iterrows():
                try:
                    # Extract feature values
                    feature_vector = [row.get(feature, 0) for feature in self.feature_names]
                    
                    # Make prediction
                    prediction = self.model.predict([feature_vector])[0]
                    
                    # Calculate confidence
                    confidence_info = self.calculate_prediction_confidence(prediction, row)
                    
                    predictions.append(prediction)
                    confidence_scores.append(confidence_info['confidence_score'])
                    confidence_levels.append(confidence_info['confidence_level'])
                    
                    batch_info['successful_predictions'] += 1
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for row {idx}: {e}")
                    predictions.append(np.nan)
                    confidence_scores.append(0.0)
                    confidence_levels.append('error')
                    batch_info['failed_predictions'] += 1
            
            # Add prediction results to DataFrame
            results_df[PREDICTION_COLUMN] = predictions
            results_df['confidence_score'] = confidence_scores
            results_df['confidence_level'] = confidence_levels
            
            batch_info['end_time'] = datetime.now()
            batch_info['processing_duration'] = (
                batch_info['end_time'] - batch_info['start_time']
            ).total_seconds()
            
            logger.info(f"Batch prediction completed: {batch_info['successful_predictions']}/{batch_info['input_records']} successful")
            
            return results_df, batch_info
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            batch_info['errors'].append(f"Batch processing error: {str(e)}")
            return df, batch_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_model_loaded():
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_type': self.model_info.get('model_type', 'Unknown'),
            'loaded_at': self.model_info.get('loaded_at'),
            'feature_count': len(self.feature_names),
            'features': self.feature_names.copy()
        }
    
    def predict(self, data: Union[pd.DataFrame, pd.Series, Dict]) -> Dict[str, Any]:
        """
        Main prediction method that handles both single and batch predictions.
        
        Args:
            data: Input data for prediction (DataFrame for batch, Series/Dict for single)
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if isinstance(data, pd.DataFrame):
                if len(data) == 1:
                    # Single row DataFrame - treat as single prediction
                    row_data = data.iloc[0]
                    return self.predict_single(row_data)
                else:
                    # Multiple rows - batch prediction
                    results_df, batch_info = self.predict_batch(data)
                    return {
                        'success': True,
                        'predictions': results_df[PREDICTION_COLUMN].tolist() if PREDICTION_COLUMN in results_df.columns else [],
                        'batch_info': batch_info,
                        'results_dataframe': results_df
                    }
            else:
                # Single prediction for Series or Dict
                return self.predict_single(data)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'confidence': None
            }

    def predict_with_explanation(self, feature_data: Union[pd.Series, Dict]) -> Dict[str, Any]:
        """
        Make prediction with detailed explanation.
        
        Args:
            feature_data: Feature values for prediction
            
        Returns:
            Dictionary containing prediction and explanation
        """
        try:
            # Get basic prediction
            prediction_result = self.predict_single(feature_data)
            
            if not prediction_result['success']:
                return prediction_result
            
            # Add explanation
            if isinstance(feature_data, dict):
                feature_data = pd.Series(feature_data)
            
            explanation = {
                'key_factors': [],
                'feature_impacts': {},
                'prediction_breakdown': {}
            }
            
            # Analyze key factors
            distance = feature_data.get('距離', 0)
            if distance > 0:
                explanation['key_factors'].append(f"レース距離: {distance}m")
            
            track_type = feature_data.get('芝・ダ_encoded', -1)
            if track_type >= 0:
                track_name = "芝" if track_type == 0 else "ダート"
                explanation['key_factors'].append(f"コース: {track_name}")
            
            venue = feature_data.get('場所_encoded', -1)
            if venue >= 0:
                explanation['key_factors'].append(f"競馬場コード: {venue}")
            
            # Add feature importance (simplified)
            for feature in ['距離', '馬番', '場所_encoded', '芝・ダ_encoded']:
                value = feature_data.get(feature, 0)
                explanation['feature_impacts'][feature] = value
            
            prediction_result['explanation'] = explanation
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction with explanation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None,
                'explanation': None
            }


def create_prediction_engine() -> PredictionEngine:
    """
    Create and return a prediction engine instance.
    
    Returns:
        Configured PredictionEngine instance
    """
    return PredictionEngine()
