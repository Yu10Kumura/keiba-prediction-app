#!/usr/bin/env python3
"""
競馬Sim V3 Phase 1統合版 起動スクリプト

V3の新機能:
- Phase 1改善版脚質推定 (順位相関0.967達成)
- レース内順位予測機能
- レース展開分析
- 騎手・調教師統計
- V2との切り替え可能
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """V3統合版アプリケーションのメイン起動処理"""
    
    # Streamlit configuration
    st.set_page_config(
        page_title="競馬走破タイム予測システム V3",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    print("🚀 競馬Sim V3 Phase 1統合版を起動中...")
    print(f"📂 作業ディレクトリ: {current_dir}")
    
    try:
        # Import and run main application
        from app import HorseRacingApp
        
        # Create and run application
        app = HorseRacingApp()
        app.run()
        
    except ImportError as e:
        st.error(f"❌ アプリケーションの読み込みに失敗しました: {e}")
        st.info("必要なモジュールが不足している可能性があります。")
        
    except Exception as e:
        st.error(f"❌ アプリケーションの実行でエラーが発生しました: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()