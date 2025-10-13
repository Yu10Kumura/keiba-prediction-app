"""
Manual input component for single race data entry.

This module provides Streamlit components for manual data entry,
form validation, and single race prediction.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, Any
import logging

from ..constants.validation_rules import VALID_VENUES
from ..core.bloodline_manager import BloodlineManager
from ..core.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class ManualInputComponent:
    """
    Handles manual data entry for single race prediction.
    
    Provides form interface for entering race data manually
    and validates the input before processing.
    """
    
    def __init__(self):
        """Initialize the manual input component."""
        from pathlib import Path
        # Get the bloodline master file path
        master_file_path = Path(__file__).parent.parent.parent / "data" / "bloodline_master.csv"
        self.bloodline_manager = BloodlineManager(str(master_file_path))
        self.data_processor = DataProcessor()
    
    def render_manual_input_form(self) -> Optional[pd.DataFrame]:
        """
        Render manual input form for race data.
        
        Returns:
            DataFrame with single race data if form is submitted, None otherwise
        """
        st.subheader("✏️ 手動データ入力")
        st.info("🎯 予測したいレースの基本情報を入力してください。AIが走破タイムを予測します。")
        
        with st.form("manual_input_form"):
            st.write("**レース基本情報**")
            
            # Basic race information
            col1, col2 = st.columns(2)
            
            with col1:
                race_date = st.date_input(
                    "開催日",
                    value=date.today(),
                    min_value=date(1990, 1, 1),
                    max_value=date(2030, 12, 31)
                )
                
                venue = st.selectbox(
                    "競馬場",
                    options=VALID_VENUES,
                    index=4  # Default to Tokyo
                )
                
                distance = st.number_input(
                    "距離 (m)",
                    min_value=800,
                    max_value=4000,
                    value=1600,
                    step=100
                )
            
            with col2:
                track_type = st.selectbox(
                    "コース種別",
                    options=['芝', 'ダ'],
                    index=0
                )
                
                track_condition = st.selectbox(
                    "馬場状態",
                    options=['良', '稍重', '重', '不良'],
                    index=0
                )
                
                horse_number = st.number_input(
                    "馬番",
                    min_value=1,
                    max_value=18,
                    value=1,
                    step=1
                )
            
            st.write("**馬情報（オプション）**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                horse_name = st.text_input(
                    "馬名",
                    placeholder="例: ディープインパクト"
                )
                
                father_name = st.text_input(
                    "父馬名",
                    placeholder="例: サンデーサイレンス"
                )
            
            with col4:
                mother_father_name = st.text_input(
                    "母の父馬名",
                    placeholder="例: ノーザンテースト"
                )
            
            # Submit button
            submitted = st.form_submit_button("データを作成", type="primary")
            
            if submitted:
                # Validate inputs
                validation_errors = self._validate_form_inputs(
                    race_date, venue, distance, track_type, 
                    track_condition, horse_number
                )
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(f"❌ {error}")
                    return None
                
                # Create DataFrame
                race_data = {
                    '年月日': race_date.strftime('%Y/%m/%d'),
                    '場所': venue,
                    '距離': distance,
                    '芝・ダ': track_type,
                    '馬場状態': track_condition,
                    '馬番': horse_number
                }
                
                # Add optional fields if provided
                if horse_name.strip():
                    race_data['馬名'] = horse_name.strip()
                
                if father_name.strip():
                    race_data['父馬名'] = father_name.strip()
                
                if mother_father_name.strip():
                    race_data['母の父馬名'] = mother_father_name.strip()
                
                df = pd.DataFrame([race_data])
                
                st.success("✅ データが作成されました")
                st.dataframe(df, use_container_width=True)
                
                return df
        
        return None
    
    def _validate_form_inputs(
        self, 
        race_date: date,
        venue: str,
        distance: int,
        track_type: str,
        track_condition: str,
        horse_number: int
    ) -> list:
        """
        Validate form inputs.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Date validation
        if race_date < date(1990, 1, 1) or race_date > date(2030, 12, 31):
            errors.append("開催日は1990年1月1日から2030年12月31日の間で入力してください")
        
        # Venue validation
        if venue not in VALID_VENUES:
            errors.append(f"競馬場は有効な値を選択してください: {venue}")
        
        # Distance validation
        if not (800 <= distance <= 4000):
            errors.append("距離は800m～4000mの間で入力してください")
        
        # Track type validation
        if track_type not in ['芝', 'ダ']:
            errors.append("コース種別は「芝」または「ダ」を選択してください")
        
        # Track condition validation
        if track_condition not in ['良', '稍重', '重', '不良']:
            errors.append("馬場状態は有効な値を選択してください")
        
        # Horse number validation
        if not (1 <= horse_number <= 18):
            errors.append("馬番は1～18の間で入力してください")
        
        return errors
    
    def render_bloodline_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Render bloodline enrichment interface.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with bloodline information added
        """
        st.subheader("🧬 血統情報の自動付与")
        
        if '父馬名' in df.columns and '母の父馬名' in df.columns:
            father_name = df.iloc[0]['父馬名'] if pd.notna(df.iloc[0]['父馬名']) else ""
            mother_father_name = df.iloc[0]['母の父馬名'] if pd.notna(df.iloc[0]['母の父馬名']) else ""
            
            if father_name or mother_father_name:
                st.write("**血統情報検索結果:**")
                
                # Show bloodline lookup results
                if father_name:
                    try:
                        father_bloodline = self.bloodline_manager.lookup_bloodline(father_name)
                        if isinstance(father_bloodline, tuple) and len(father_bloodline) == 2:
                            small_lineage, country_lineage = father_bloodline
                            if small_lineage != 'UNK' and country_lineage != 'UNK':
                                st.write(f"🔍 父馬「{father_name}」:")
                                st.write(f"  小系統: {small_lineage}")
                                st.write(f"  国系統: {country_lineage}")
                            else:
                                st.write(f"⚠️ 父馬「{father_name}」の血統情報が見つかりませんでした")
                        else:
                            st.write(f"⚠️ 父馬「{father_name}」の血統情報が見つかりませんでした")
                    except Exception as e:
                        st.error(f"血統検索エラー（父馬）: {str(e)}")
                        logger.error(f"Father bloodline lookup error: {e}")
                
                if mother_father_name:
                    try:
                        mother_father_bloodline = self.bloodline_manager.lookup_bloodline(mother_father_name)
                        if isinstance(mother_father_bloodline, tuple) and len(mother_father_bloodline) == 2:
                            small_lineage, country_lineage = mother_father_bloodline
                            if small_lineage != 'UNK' and country_lineage != 'UNK':
                                st.write(f"🔍 母の父馬「{mother_father_name}」:")
                                st.write(f"  小系統: {small_lineage}")
                                st.write(f"  国系統: {country_lineage}")
                            else:
                                st.write(f"⚠️ 母の父馬「{mother_father_name}」の血統情報が見つかりませんでした")
                        else:
                            st.write(f"⚠️ 母の父馬「{mother_father_name}」の血統情報が見つかりませんでした")
                    except Exception as e:
                        st.error(f"血統検索エラー（母の父馬）: {str(e)}")
                        logger.error(f"Mother father bloodline lookup error: {e}")
                
                # Apply bloodline enrichment
                try:
                    enriched_df = self.bloodline_manager.enrich_dataframe(df)
                    
                    st.write("**血統情報付与後のデータ:**")
                    st.dataframe(enriched_df, use_container_width=True)
                    
                    # Automatically process the data after bloodline enrichment
                    st.write("**データ前処理中...**")
                    with st.spinner("前処理を実行中..."):
                        try:
                            processed_df, processing_info = self.data_processor.process_data(enriched_df)
                            
                            if processing_info['feature_engineering_completed']:
                                st.success("✅ 前処理が完了しました")
                                
                                # Show processing summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("処理前の列数", len(enriched_df.columns))
                                    st.metric("処理後の列数", len(processed_df.columns))
                                
                                with col2:
                                    duration = processing_info.get('processing_duration', 0)
                                    st.metric("処理時間", f"{duration:.3f}秒")
                                
                                return processed_df
                            else:
                                st.error("❌ 前処理に失敗しました")
                                if processing_info['errors']:
                                    for error in processing_info['errors']:
                                        st.write(f"• {error}")
                        
                        except Exception as e:
                            st.error(f"❌ 前処理エラー: {str(e)}")
                            logger.error(f"Manual input preprocessing error: {e}")
                    
                    return enriched_df
                    
                except Exception as e:
                    st.error(f"❌ 血統情報付与エラー: {str(e)}")
                    logger.error(f"Bloodline enrichment error: {e}")
                    
                    # エラーの場合は元のデータで前処理を実行
                    st.write("**血統情報なしでデータ前処理中...**")
                    with st.spinner("前処理を実行中..."):
                        try:
                            processed_df, processing_info = self.data_processor.process_data(df)
                            
                            if processing_info['feature_engineering_completed']:
                                st.success("✅ 前処理が完了しました")
                                
                                # Show processing summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("処理前の列数", len(df.columns))
                                    st.metric("処理後の列数", len(processed_df.columns))
                                
                                with col2:
                                    duration = processing_info.get('processing_duration', 0)
                                    st.metric("処理時間", f"{duration:.3f}秒")
                                
                                return processed_df
                            else:
                                st.error("❌ 前処理に失敗しました")
                                if processing_info['errors']:
                                    for error in processing_info['errors']:
                                        st.write(f"• {error}")
                        
                        except Exception as e:
                            st.error(f"❌ 前処理エラー: {str(e)}")
                            logger.error(f"Manual input preprocessing error: {e}")
                    
                    return df
            else:
                st.info("血統情報が入力されていないため、血統付与をスキップします")
                
                # Process data without bloodline information
                st.write("**データ前処理中...**")
                with st.spinner("前処理を実行中..."):
                    try:
                        processed_df, processing_info = self.data_processor.process_data(df)
                        
                        if processing_info['feature_engineering_completed']:
                            st.success("✅ 前処理が完了しました")
                            
                            # Show processing summary
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("処理前の列数", len(df.columns))
                                st.metric("処理後の列数", len(processed_df.columns))
                            
                            with col2:
                                duration = processing_info.get('processing_duration', 0)
                                st.metric("処理時間", f"{duration:.3f}秒")
                            
                            return processed_df
                        else:
                            st.error("❌ 前処理に失敗しました")
                            if processing_info['errors']:
                                for error in processing_info['errors']:
                                    st.write(f"• {error}")
                    
                    except Exception as e:
                        st.error(f"❌ 前処理エラー: {str(e)}")
                        logger.error(f"Manual input preprocessing error: {e}")
        else:
            st.info("血統情報列がないため、血統付与をスキップします")
            
            # Process data without bloodline information
            st.write("**データ前処理中...**")
            with st.spinner("前処理を実行中..."):
                try:
                    processed_df, processing_info = self.data_processor.process_data(df)
                    
                    if processing_info['feature_engineering_completed']:
                        st.success("✅ 前処理が完了しました")
                        
                        # Show processing summary
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("処理前の列数", len(df.columns))
                            st.metric("処理後の列数", len(processed_df.columns))
                        
                        with col2:
                            duration = processing_info.get('processing_duration', 0)
                            st.metric("処理時間", f"{duration:.3f}秒")
                        
                        return processed_df
                    else:
                        st.error("❌ 前処理に失敗しました")
                        if processing_info['errors']:
                            for error in processing_info['errors']:
                                st.write(f"• {error}")
                
                except Exception as e:
                    st.error(f"❌ 前処理エラー: {str(e)}")
                    logger.error(f"Manual input preprocessing error: {e}")
        
        return df
    
    def render_input_summary(self, df: pd.DataFrame) -> None:
        """
        Render input data summary.
        
        Args:
            df: Input DataFrame to summarize
        """
        st.subheader("📋 入力データサマリー")
        
        if len(df) > 0:
            race_data = df.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**レース情報:**")
                st.write(f"📅 開催日: {race_data.get('年月日', 'N/A')}")
                st.write(f"🏟️ 競馬場: {race_data.get('場所', 'N/A')}")
                st.write(f"📏 距離: {race_data.get('距離', 'N/A')}m")
                st.write(f"🌱 コース: {race_data.get('芝・ダ', 'N/A')}")
                st.write(f"🌧️ 馬場状態: {race_data.get('馬場状態', 'N/A')}")
            
            with col2:
                st.write("**馬情報:**")
                st.write(f"🏇 馬番: {race_data.get('馬番', 'N/A')}")
                st.write(f"🐎 馬名: {race_data.get('馬名', '未入力')}")
                st.write(f"👨 父馬: {race_data.get('父馬名', '未入力')}")
                st.write(f"👩 母の父: {race_data.get('母の父馬名', '未入力')}")
        else:
            st.warning("表示するデータがありません")
