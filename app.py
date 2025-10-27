"""
Main Streamlit application for horse racing time prediction.

This is the entry point for the web application that provides
both file upload and manual input interfaces for race prediction.
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Configure page
st.set_page_config(
    page_title="競馬走破タイム予測システム",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
try:
    from src.components.data_input import DataInputComponent
    from src.components.manual_input import ManualInputComponent
    from src.components.result_display import ResultDisplayComponent
    from src.core.prediction_engine import PredictionEngine
    from src.utils.config_manager import ConfigManager
except Exception as e:
    st.error(f"❌ コンポーネントのインポートエラー: {str(e)}")
    st.stop()
logger = logging.getLogger(__name__)


class HorseRacingApp:
    """
    Main application class for horse racing prediction system.
    
    Handles the overall application flow, UI management,
    and coordination between different components.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.config_manager = ConfigManager()
        # Initialize components as None - they will be created when needed
        self.data_input_component = None
        self.manual_input_component = None
        self.result_display_component = None
        self.prediction_engine = None
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load prediction engine
        self._load_prediction_engine()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = None
        
        if 'input_method' not in st.session_state:
            st.session_state.input_method = "ファイル"
        
        if 'show_advanced' not in st.session_state:
            st.session_state.show_advanced = False
    
    def _load_prediction_engine(self):
        """Load the prediction engine."""
        try:
            self.prediction_engine = PredictionEngine()
            logger.info("Prediction engine loaded successfully")
        except Exception as e:
            st.error(f"❌ 予測エンジンの読み込みに失敗しました: {str(e)}")
            logger.error(f"Failed to load prediction engine: {e}")
    
    def _get_data_input_component(self):
        """Get or create data input component."""
        if self.data_input_component is None:
            self.data_input_component = DataInputComponent()
        return self.data_input_component
    
    def _get_manual_input_component(self):
        """Get or create manual input component."""
        if self.manual_input_component is None:
            self.manual_input_component = ManualInputComponent()
        return self.manual_input_component
    
    def _get_result_display_component(self):
        """Get or create result display component."""
        if self.result_display_component is None:
            self.result_display_component = ResultDisplayComponent()
        return self.result_display_component
    
    def run(self):
        """Run the main application."""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
        self._render_footer()
    
    def _render_header(self):
        """Render the application header."""
        st.title("🏇 競馬走破タイム予測システム")
        st.markdown("""
        **機械学習を使用して競馬の走破タイムを予測するシステムです。**  
        
        レースの基本情報（日付、競馬場、距離、馬場状態など）を入力すると、AI（LightGBM）が走破タイムを予測します。  
        CSVファイルのアップロードまたは手動入力でレースデータを入力できます。
        """)
        
        # System status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            engine_status = "🟢 正常" if self.prediction_engine else "🔴 エラー"
            st.metric("予測エンジン", engine_status)
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        with st.sidebar:
            st.header("⚙️ 設定")
            
            # Input method selection
            st.session_state.input_method = st.radio(
                "入力方法を選択",
                ["ファイル", "手動入力"],
                index=0 if st.session_state.input_method == "ファイル" else 1
            )
            
            st.divider()
            
            # Advanced options
            st.session_state.show_advanced = st.checkbox(
                "詳細オプション",
                value=st.session_state.show_advanced
            )
            
            if st.session_state.show_advanced:
                st.subheader("詳細設定")
                
                # Prediction confidence threshold
                confidence_threshold = st.slider(
                    "信頼度閾値",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=1.0,
                    help="この値以下の予測は警告を表示"
                )
                
                # Batch processing options
                st.checkbox("バッチ処理モード", value=False)
                st.checkbox("詳細ログ出力", value=False)
            
            st.divider()
            
            # Clear data button
            if st.button("データをクリア", type="secondary"):
                self._clear_session_data()
                st.rerun()
    
    def _render_main_content(self):
        """Render the main content area."""
        
        # Input section
        if st.session_state.input_method == "ファイル":
            self._render_file_input_section()
        else:
            self._render_manual_input_section()
        
        # Prediction section
        if st.session_state.processed_data is not None:
            self._render_prediction_section()
        
        # Results section
        if st.session_state.prediction_results is not None:
            self._render_results_section()
    
    def _render_file_input_section(self):
        """Render file input section."""
        st.header("📁 ファイル入力")
        
        # Get component instance
        data_input_component = self._get_data_input_component()
        
        # File upload
        uploaded_data = data_input_component.render_file_upload()
        
        if uploaded_data is not None:
            # Data validation
            data_input_component.render_data_validation(uploaded_data)
            
            # Skip column mapping if CSV already has correct column names
            # Direct data processing
            processed_data = data_input_component.render_data_processing(uploaded_data)
            
            if processed_data is not None:
                st.session_state.processed_data = processed_data
                st.success("✅ データ処理が完了しました")
                st.rerun()
                
                if processed_data is not None:
                    st.session_state.processed_data = processed_data
                    st.success("✅ データ処理が完了しました")
                    st.rerun()
    
    def _render_manual_input_section(self):
        """Render manual input section."""
        st.header("✏️ 手動入力")
        
        # Get component instance
        manual_input_component = self._get_manual_input_component()
        
        # Manual input form
        manual_data = manual_input_component.render_manual_input_form()
        
        if manual_data is not None:
            # Input summary
            manual_input_component.render_input_summary(manual_data)
            
            # Bloodline enrichment and automatic data processing
            processed_data = manual_input_component.render_bloodline_enrichment(manual_data)
            
            if processed_data is not None:
                st.session_state.processed_data = processed_data
                st.success("✅ データ処理が完了しました")
                st.rerun()
    
    def _render_prediction_section(self):
        """Render prediction section."""
        st.header("🎯 予測実行")
        
        processed_data = st.session_state.processed_data
        
        if processed_data is None:
            st.warning("処理済みデータがありません")
            return
        
        # Show processed data summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("データ行数", len(processed_data))
        with col2:
            st.metric("特徴量数", len(processed_data.columns))
        
        # Prediction execution
        if st.button("予測を実行", type="primary", disabled=self.prediction_engine is None):
            if self.prediction_engine is None:
                st.error("❌ 予測エンジンが利用できません")
                return
            
            with st.spinner("予測を実行中..."):
                try:
                    # Debug: Show processed data info
                    st.write(f"DEBUG: 処理済みデータ形状: {processed_data.shape}")
                    st.write(f"DEBUG: 処理済みデータ列: {processed_data.columns.tolist()}")
                    
                    prediction_results = self.prediction_engine.predict(processed_data)
                    
                    # Debug: Show prediction results structure
                    st.write(f"DEBUG: 予測結果の型: {type(prediction_results)}")
                    st.write(f"DEBUG: 予測結果の内容: {prediction_results}")
                    
                    st.session_state.prediction_results = prediction_results
                    st.success("✅ 予測が完了しました")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ 予測エラー: {str(e)}")
                    logger.error(f"Prediction error: {e}")
                    import traceback
                    st.write(f"DEBUG: エラー詳細: {traceback.format_exc()}")
        
        # Show sample of processed data
        with st.expander("処理済みデータプレビュー", expanded=False):
            st.dataframe(processed_data.head(), use_container_width=True)
    
    def _render_results_section(self):
        """Render results section."""
        st.header("📊 予測結果")
        
        prediction_results = st.session_state.prediction_results
        input_data = st.session_state.processed_data
        
        if prediction_results is None:
            st.warning("表示する予測結果がありません")
            return
        
        # Get component instance
        result_display_component = self._get_result_display_component()
        
        # Single prediction results
        if isinstance(prediction_results, dict):
            result_display_component.render_prediction_results(
                prediction_results, input_data
            )
        
        # Batch prediction results
        elif isinstance(prediction_results, list):
            result_display_component.render_batch_results(prediction_results)
        
        # Export results option
        self._render_export_section(prediction_results)
    
    def _render_export_section(self, results):
        """Render export options for results."""
        with st.expander("💾 結果エクスポート", expanded=False):
            st.write("予測結果を保存できます。")
            
            if isinstance(results, dict) and 'predictions' in results:
                # Single prediction export
                export_data = {
                    'predicted_time': results['predictions'][0],
                    'confidence': results.get('confidence', 0),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                if st.button("JSON形式でダウンロード"):
                    st.download_button(
                        label="結果をダウンロード",
                        data=pd.Series(export_data).to_json(indent=2),
                        file_name=f"prediction_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    def _render_footer(self):
        """Render the application footer."""
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**対応データ:**")
            st.write("• CSVファイル")
            st.write("• 手動入力")
        
        with col2:
            st.write("**入力項目:**")
            st.write("• レース基本情報")
            st.write("• 血統情報（オプション）")
        
        with col3:
            st.write("**出力結果:**")
            st.write("• 走破タイム予測")
            st.write("• 信頼度表示")
    
    def _clear_session_data(self):
        """Clear all session data."""
        st.session_state.processed_data = None
        st.session_state.prediction_results = None
        logger.info("Session data cleared")


def main():
    """Main application entry point."""
    try:
        app = HorseRacingApp()
        app.run()
    except Exception as e:
        st.error(f"❌ アプリケーションエラー: {str(e)}")
        logger.error(f"Application error: {e}")
        
        with st.expander("エラー詳細", expanded=False):
            st.code(str(e), language="text")


if __name__ == "__main__":
    main()
