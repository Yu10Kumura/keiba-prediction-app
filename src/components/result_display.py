"""
Result display component for prediction visualization.

This module provides Streamlit components for displaying prediction results,
performance metrics, and comparative analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ResultDisplayComponent:
    """
    Handles display of prediction results and performance metrics.
    
    Provides visualization components for prediction results,
    confidence intervals, and comparative analysis.
    """
    
    def render_prediction_results(
        self, 
        predictions: Dict[str, Any],
        input_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Render prediction results with metrics and visualization.
        
        Args:
            predictions: Dictionary containing prediction results
            input_data: Original input data for context
        """
        st.subheader("🎯 予測結果")
        
    def render_prediction_results(
        self, 
        predictions: Dict[str, Any],
        input_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Render prediction results with metrics and visualization.
        
        Args:
            predictions: Dictionary containing prediction results
            input_data: Original input data for context
        """
        st.subheader("🎯 予測結果")
        
        if not predictions:
            st.error("❌ 予測結果がありません")
            return
        
        # Handle different prediction result formats
        if isinstance(predictions, dict):
            if 'success' in predictions and not predictions['success']:
                st.error(f"❌ 予測に失敗しました: {predictions.get('error', '不明なエラー')}")
                return
            
            # Get predictions array
            prediction_values = None
            if 'predictions' in predictions:
                prediction_values = predictions['predictions']
            elif 'prediction' in predictions:
                prediction_values = predictions['prediction']
                if not isinstance(prediction_values, list):
                    prediction_values = [prediction_values]
            else:
                st.error("❌ 予測結果が見つかりません")
                return
        else:
            st.error("❌ 予測結果の形式が不正です")
            return
        
        # Create results DataFrame
        horse_names = []
        if input_data is not None and len(input_data) > 0:
            try:
                # G列（インデックス6）から馬名を直接取得
                if input_data.shape[1] > 6:
                    horse_names = input_data.iloc[:, 6].astype(str).tolist()
                    # 長さを予測結果に合わせる
                    horse_names = horse_names[:len(prediction_values)]
            except Exception:
                horse_names = []
        
        # If no horse names found, create default names
        if not horse_names or len(horse_names) != len(prediction_values):
            horse_names = [f"馬{i}" for i in range(1, len(prediction_values) + 1)]
        
        results_df = pd.DataFrame({
            '馬番': range(1, len(prediction_values) + 1),
            '馬名': horse_names,
            '予測走破タイム(秒)': [f"{pred:.2f}" for pred in prediction_values],
            '順位予想': range(1, len(prediction_values) + 1)  # Will be sorted by time
        })
        
        # Sort by predicted time to get ranking
        results_df['予測走破タイム_数値'] = prediction_values
        results_df = results_df.sort_values('予測走破タイム_数値').reset_index(drop=True)
        results_df['順位予想'] = range(1, len(results_df) + 1)
        results_df = results_df.drop('予測走破タイム_数値', axis=1)
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fastest_time = min(prediction_values)
            st.metric("最速予想タイム", f"{fastest_time:.2f}秒")
        
        with col2:
            slowest_time = max(prediction_values)
            st.metric("最遅予想タイム", f"{slowest_time:.2f}秒")
        
        with col3:
            time_range = slowest_time - fastest_time
            st.metric("タイム幅", f"{time_range:.2f}秒")
        
        # Display results table
        st.write("**🏇 全馬予測結果**")
        st.dataframe(
            results_df,
            width='stretch',
            hide_index=True,
            column_config={
                "馬番": st.column_config.NumberColumn("馬番", width="small"),
                "馬名": st.column_config.TextColumn("馬名", width="medium"),
                "予測走破タイム(秒)": st.column_config.TextColumn("予測走破タイム(秒)", width="medium"),
                "順位予想": st.column_config.NumberColumn("順位予想", width="small")
            }
        )
        
        # Add context if input data is available
        if input_data is not None and len(input_data) > 0:
            st.write("**📊 入力データ概要**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("対象レース頭数", len(input_data))
            with col2:
                if '距離' in input_data.columns:
                    distance = input_data['距離'].iloc[0] if not input_data['距離'].empty else "不明"
                    st.metric("レース距離", f"{distance}m")
        
        # Feature importance if available
        if 'feature_importance' in predictions:
            self._render_feature_importance(predictions['feature_importance'])
    
    def _render_prediction_details(
        self, 
        predictions: Dict[str, Any], 
        input_data: Optional[pd.DataFrame]
    ) -> None:
        """Render detailed prediction information."""
        
        with st.expander("📊 予測詳細", expanded=False):
            pred_info = predictions.get('prediction_info', {})
            
            if pred_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**予測情報:**")
                    st.write(f"• 使用特徴量数: {pred_info.get('num_features', 'N/A')}")
                    st.write(f"• 処理時間: {pred_info.get('processing_time', 'N/A'):.3f}秒")
                    st.write(f"• 予測時間: {pred_info.get('prediction_time', 'N/A'):.3f}秒")
                
                with col2:
                    st.write("**モデル情報:**")
                    model_info = predictions.get('model_info', {})
                    st.write(f"• モデル種別: {model_info.get('type', 'LightGBM')}")
                    st.write(f"• 学習データ数: {model_info.get('n_training_samples', 'N/A')}")
                    st.write(f"• 特徴量数: {model_info.get('n_features', 'N/A')}")
            
            # Show input race conditions
            if input_data is not None and len(input_data) > 0:
                st.write("**レース条件:**")
                race_data = input_data.iloc[0]
                
                conditions = []
                if '場所' in race_data:
                    conditions.append(f"競馬場: {race_data['場所']}")
                if '距離' in race_data:
                    conditions.append(f"距離: {race_data['距離']}m")
                if '芝・ダ' in race_data:
                    conditions.append(f"コース: {race_data['芝・ダ']}")
                if '馬場状態' in race_data:
                    conditions.append(f"馬場: {race_data['馬場状態']}")
                
                for condition in conditions:
                    st.write(f"• {condition}")
    
    def _render_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """Render feature importance visualization."""
        
        with st.expander("🎯 特徴量重要度", expanded=False):
            if not feature_importance:
                st.write("特徴量重要度情報が利用できません")
                return
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame([
                {'特徴量': feature, '重要度': importance}
                for feature, importance in feature_importance.items()
            ]).sort_values('重要度', ascending=True)
            
            # Plot horizontal bar chart
            fig = px.bar(
                importance_df,
                x='重要度',
                y='特徴量',
                orientation='h',
                title='特徴量重要度',
                labels={'重要度': '重要度', '特徴量': '特徴量'}
            )
            
            fig.update_layout(
                height=max(400, len(importance_df) * 25),
                margin=dict(l=150, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features table
            st.write("**上位特徴量:**")
            top_features = importance_df.tail(10).sort_values('重要度', ascending=False)
            st.dataframe(top_features, use_container_width=True, hide_index=True)
    
    def _render_time_comparison(self, predicted_time: float, actual_time: float) -> None:
        """Render comparison between predicted and actual times."""
        
        st.subheader("⚖️ 実績タイムとの比較")
        
        difference = predicted_time - actual_time
        accuracy_pct = (1 - abs(difference) / actual_time) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "実際の走破タイム",
                f"{actual_time:.2f}秒"
            )
        
        with col2:
            delta_color = "inverse" if difference > 0 else "normal"
            st.metric(
                "予測との差",
                f"{difference:+.2f}秒",
                delta=f"{difference:+.2f}秒"
            )
        
        with col3:
            st.metric(
                "予測精度",
                f"{accuracy_pct:.1f}%"
            )
        
        # Visualization
        times_df = pd.DataFrame({
            'タイプ': ['実際', '予測'],
            '走破タイム': [actual_time, predicted_time]
        })
        
        fig = px.bar(
            times_df,
            x='タイプ',
            y='走破タイム',
            title='走破タイム比較',
            color='タイプ',
            color_discrete_map={'実際': '#ff6b6b', '予測': '#4ecdc4'}
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy assessment
        if abs(difference) < 1.0:
            st.success(f"✅ 高精度予測（誤差: {abs(difference):.2f}秒）")
        elif abs(difference) < 3.0:
            st.info(f"ℹ️ 良好な予測（誤差: {abs(difference):.2f}秒）")
        else:
            st.warning(f"⚠️ 予測誤差が大きいです（誤差: {abs(difference):.2f}秒）")
    
    def render_batch_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Render results for batch predictions.
        
        Args:
            results: List of prediction result dictionaries
        """
        st.subheader("📊 バッチ予測結果")
        
        if not results:
            st.warning("表示する結果がありません")
            return
        
        # Create summary DataFrame
        summary_data = []
        for i, result in enumerate(results):
            if 'predictions' in result and len(result['predictions']) > 0:
                predicted_time = result['predictions'][0]
                confidence = result.get('confidence', 0)
                
                summary_data.append({
                    'レース番号': i + 1,
                    '予測タイム': f"{predicted_time:.2f}秒",
                    '信頼度': f"{confidence:.1f}%",
                    '予測値': predicted_time  # For sorting
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Display summary table
            display_df = summary_df.drop('予測値', axis=1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            predicted_times = [d['予測値'] for d in summary_data]
            
            with col1:
                st.metric("予測数", len(predicted_times))
            
            with col2:
                st.metric("平均タイム", f"{sum(predicted_times)/len(predicted_times):.2f}秒")
            
            with col3:
                st.metric("最速タイム", f"{min(predicted_times):.2f}秒")
            
            with col4:
                st.metric("最遅タイム", f"{max(predicted_times):.2f}秒")
            
            # Distribution visualization
            fig = px.histogram(
                x=predicted_times,
                nbins=20,
                title='予測タイム分布',
                labels={'x': '予測タイム（秒）', 'y': '頻度'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_error_message(self, error_message: str, details: Optional[str] = None) -> None:
        """
        Render error message with optional details.
        
        Args:
            error_message: Main error message
            details: Optional detailed error information
        """
        st.error(f"❌ {error_message}")
        
        if details:
            with st.expander("エラー詳細", expanded=False):
                st.code(details, language="text")
    
    def render_success_message(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Render success message with optional details.
        
        Args:
            message: Success message
            details: Optional additional information
        """
        st.success(f"✅ {message}")
        
        if details:
            with st.expander("処理詳細", expanded=False):
                for key, value in details.items():
                    st.write(f"**{key}:** {value}")
    
    def render_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Render model performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        st.subheader("📈 モデル性能指標")
        
        if not metrics:
            st.info("性能指標が利用できません")
            return
        
        # Display metrics in columns
        metric_cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with metric_cols[i]:
                if isinstance(metric_value, float):
                    st.metric(metric_name, f"{metric_value:.4f}")
                else:
                    st.metric(metric_name, str(metric_value))
        
        # Performance assessment
        if 'rmse' in metrics:
            rmse = metrics['rmse']
            if rmse < 2.0:
                st.success("🎯 優秀なモデル性能です")
            elif rmse < 5.0:
                st.info("📊 良好なモデル性能です")
            else:
                st.warning("⚠️ モデル性能の改善が必要です")
