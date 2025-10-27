"""
Data input component for file upload and CSV processing.

This module provides Streamlit components for file upload,
data validation, and initial data processing display.
"""

import streamlit as st
import pandas as pd
import io
from typing import Optional, Tuple, Dict, Any
import logging

from ..constants.validation_rules import FILE_CONSTRAINTS, VALID_VENUES
from ..constants.messages import UI_MESSAGES
from ..core.bloodline_manager import BloodlineManager
from ..core.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DataInputComponent:
    """
    Handles file upload and data input functionality.
    
    Provides file upload interface, data validation display,
    and initial data processing for CSV files.
    """
    
    def __init__(self):
        """Initialize the data input component."""
        from pathlib import Path
        # Get the bloodline master file path
        master_file_path = Path(__file__).parent.parent.parent / "data" / "bloodline_master.csv"
        self.bloodline_manager = BloodlineManager(str(master_file_path))
        self.data_processor = DataProcessor()
    
    def render_file_upload(self) -> Optional[pd.DataFrame]:
        """
        Render file upload interface.
        
        Returns:
            Uploaded DataFrame or None if no valid file uploaded
        """
        st.subheader("📂 データファイルアップロード")
        
        # File upload widget
        uploaded_file = st.file_uploader(
            "CSVファイルをアップロードしてください",
            type=['csv'],
            help=f"最大ファイルサイズ: {FILE_CONSTRAINTS['max_file_size'] // (1024*1024)}MB"
        )
        
        if uploaded_file is not None:
            try:
                # File size check
                file_size = len(uploaded_file.getvalue())
                max_size = FILE_CONSTRAINTS['max_file_size']
                
                if file_size > max_size:
                    st.error(f"⚠️ ファイルサイズが大きすぎます: {file_size // (1024*1024)}MB > {max_size // (1024*1024)}MB")
                    return None
                
                # Read CSV file
                df = self._read_csv_file(uploaded_file)
                
                if df is not None:
                    st.success(f"✅ ファイル読み込み成功: {len(df)}行, {len(df.columns)}列")
                    
                    # Display data preview
                    self._display_data_preview(df)
                    
                    return df
                
            except Exception as e:
                st.error(f"❌ ファイル読み込みエラー: {str(e)}")
                logger.error(f"File upload error: {e}")
        
        return None
    
    def _read_csv_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Read CSV file with appropriate encoding.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            DataFrame or None if read failed
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'shift_jis', 'cp932']
            
            for encoding in encodings:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Read CSV
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    
                    # Basic validation
                    if len(df) == 0:
                        st.warning("⚠️ 空のファイルです")
                        return None
                    
                    if len(df) > FILE_CONSTRAINTS['max_rows']:
                        st.warning(f"⚠️ 行数が多すぎます: {len(df)} > {FILE_CONSTRAINTS['max_rows']}")
                        return df.head(FILE_CONSTRAINTS['max_rows'])
                    
                    st.info(f"📋 エンコーディング: {encoding}")
                    return df
                    
                except UnicodeDecodeError:
                    continue
                    
            st.error("❌ サポートされていないエンコーディングです")
            return None
            
        except Exception as e:
            st.error(f"❌ CSV読み込みエラー: {str(e)}")
            return None
    
    def _display_data_preview(self, df: pd.DataFrame) -> None:
        """
        Display data preview and basic information.
        
        Args:
            df: DataFrame to preview
        """
        st.subheader("📊 データプレビュー")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("行数", len(df))
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            missing_cells = df.isnull().sum().sum()
            st.metric("欠損値", missing_cells)
        
        # Column information
        st.write("**列一覧:**")
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            column_info.append({
                '列名': col,
                'データ型': dtype,
                '欠損値': null_count,
                'ユニーク値数': unique_count
            })
        
        st.dataframe(pd.DataFrame(column_info), use_container_width=True)
        
        # Data preview
        st.write("**データサンプル（最初の5行）:**")
        st.dataframe(df.head(), use_container_width=True)
    
    def render_data_validation(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Render data validation interface and results.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        st.subheader("🔍 データバリデーション")
        
        # Run validation
        is_valid, errors = self.data_processor.validate_input_data(df)
        
        # Display validation results
        if is_valid:
            st.success("✅ データバリデーション成功")
        else:
            st.error("❌ データバリデーションエラー")
            
            st.write("**エラー詳細:**")
            for error in errors:
                st.write(f"• {error}")
        
        # Display column mapping
        self._display_column_mapping(df)
        
        validation_info = {
            'is_valid': is_valid,
            'errors': errors,
            'input_records': len(df)
        }
        
        return is_valid, validation_info
    
    def _display_column_mapping(self, df: pd.DataFrame) -> None:
        """
        Display column mapping and requirements.
        
        Args:
            df: Input DataFrame
        """
        st.write("**列マッピング状況:**")
        
        required_columns = [
            '年月日', '場所', '芝・ダ', '距離', '馬場状態', '馬番'
        ]
        
        optional_columns = [
            '馬名', '父馬名', '母の父馬名'  # 走破タイムを除外
        ]
        
        # Check required columns
        st.write("*必須列:*")
        for col in required_columns:
            if col in df.columns:
                st.write(f"✅ {col}")
            else:
                st.write(f"❌ {col} (不足)")
        
        # Check optional columns
        st.write("*オプション列:*")
        for col in optional_columns:
            if col in df.columns:
                st.write(f"✅ {col}")
            else:
                st.write(f"⚪ {col} (なし)")
    
    def render_column_mapping(self, uploaded_df: pd.DataFrame) -> Dict[str, str]:
        """
        Render column mapping interface for CSV upload.
        
        Args:
            uploaded_df: Uploaded DataFrame
            
        Returns:
            Dictionary mapping CSV columns to required columns
        """
        st.subheader("📋 カラムマッピング")
        st.write("CSVファイルの列名をアプリで使用する列名にマッピングしてください。")
        
        # 必須カラム（走破タイムを除外）
        required_columns = {
            '開催日': 'year_month_day',
            '競馬場': 'venue', 
            '距離': 'distance',
            '芝・ダ': 'track_type',
            '馬場状態': 'track_condition',
            '馬番': 'horse_number',
            '父馬名': 'father_name',
            '母父馬名': 'mother_father_name'
        }
        
        csv_columns = ['選択してください'] + list(uploaded_df.columns)
        column_mapping = {}
        
        st.write("**必須項目のマッピング:**")
        for display_name, internal_name in required_columns.items():
            selected_column = st.selectbox(
                f"{display_name}:",
                options=csv_columns,
                key=f"mapping_{internal_name}"
            )
            if selected_column != '選択してください':
                column_mapping[internal_name] = selected_column
        
        return column_mapping
    
    def render_data_processing(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Render data processing interface and execute processing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed_dataframe, processing_info)
        """
        st.subheader("⚙️ データ前処理")
        
        # Initialize session state for processing results
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'processing_info' not in st.session_state:
            st.session_state.processing_info = {}
        if 'processing_completed' not in st.session_state:
            st.session_state.processing_completed = False
        
        # Show results if already processed
        if st.session_state.processing_completed and st.session_state.processed_data is not None:
            st.success("✅ データ処理が完了しました")
            
            # Display processing results
            st.write("**処理結果:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("入力レコード数", st.session_state.processing_info['input_records'])
                st.metric("出力レコード数", st.session_state.processing_info['output_records'])
            
            with col2:
                duration = st.session_state.processing_info.get('processing_duration', 0)
                st.metric("処理時間", f"{duration:.3f}秒")
                
                if st.session_state.processing_info['errors']:
                    st.metric("エラー数", len(st.session_state.processing_info['errors']))
            
            # Show processing steps status
            steps = [
                ('バリデーション', st.session_state.processing_info['validation_passed']),
                ('データクリーニング', st.session_state.processing_info['cleaning_completed']),
                ('特徴量生成', st.session_state.processing_info['feature_engineering_completed'])
            ]
            
            st.write("**処理ステップ:**")
            for step_name, status in steps:
                status_icon = "✅" if status else "❌"
                st.write(f"{status_icon} {step_name}")
            
            # Show processed data preview
            if len(st.session_state.processed_data) > 0:
                st.write("**処理後データプレビュー:**")
                st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
            
            # Reset button
            if st.button("🔄 データを再処理", type="secondary"):
                st.session_state.processing_completed = False
                st.session_state.processed_data = None
                st.session_state.processing_info = {}
                st.rerun()
                
            return st.session_state.processed_data, st.session_state.processing_info
        
        # Process button
        if st.button("データ前処理を実行", type="primary"):
            with st.spinner("処理中..."):
                # Step 1: Bloodline enrichment
                if '父馬名' in df.columns and '母の父馬名' in df.columns:
                    st.write("🧬 血統情報付与中...")
                    enriched_df = self.bloodline_manager.enrich_dataframe(df)
                    bloodline_added = len(enriched_df.columns) - len(df.columns)
                    st.write(f"✅ 血統列追加: {bloodline_added}列")
                else:
                    st.write("⚠️ 血統情報列がないため、血統付与をスキップ")
                    enriched_df = df.copy()
                
                # Step 2: Data processing
                st.write("📊 特徴量エンジニアリング中...")
                processed_df, processing_info = self.data_processor.process_data(enriched_df)
                
                # Store results in session state
                st.session_state.processed_data = processed_df
                st.session_state.processing_info = processing_info
                st.session_state.processing_completed = True
                
                st.success("✅ データ処理が完了しました")
                st.rerun()
        
        return None, {}
    
    def render_sample_data_option(self) -> Optional[pd.DataFrame]:
        """
        Render sample data option for testing.
        
        Returns:
            Sample DataFrame if selected, None otherwise
        """
        st.subheader("🧪 サンプルデータでテスト")
        
        if st.button("サンプルデータを使用"):
            from ..core.data_processor import create_sample_data
            sample_df = create_sample_data()
            
            st.success("✅ サンプルデータを読み込みました")
            st.dataframe(sample_df, use_container_width=True)
            
            return sample_df
        
        return None
