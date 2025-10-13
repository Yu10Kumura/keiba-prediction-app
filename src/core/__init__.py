"""
Core functionality package for the horse racing simulation app.

This package contains the core business logic including:
- BloodlineManager: Horse bloodline lookup and data enrichment
- DataProcessor: Comprehensive data preprocessing and feature engineering
- PredictionEngine: Machine learning prediction functionality
"""

from .bloodline_manager import BloodlineManager
from .data_processor import DataProcessor
from .prediction_engine import PredictionEngine, create_prediction_engine

__all__ = ['BloodlineManager', 'DataProcessor', 'PredictionEngine', 'create_prediction_engine']
