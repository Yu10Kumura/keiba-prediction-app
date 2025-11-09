"""
競馬予測システム V4 - Streamlit アプリケーション
レース前情報のみを使用したタイム予測システム

機能:
- CSV一括予測
- 手動入力予測  
- V4高精度モデル（MAE 0.961秒）
- レース前情報のみ使用で実用的
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List, Tuple
import traceback
import unicodedata

# ページ設定
st.set_page_config(
    page_title="🏇 競馬予測システム V4",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BloodlineManager:
    """血統マスタ管理クラス"""
    
    def __init__(self, master_file_path: str):
        """血統マスタファイルを読み込んで初期化"""
        self.master_file_path = Path(master_file_path)
        self.bloodline_dict = {}
        self.logger = logging.getLogger(__name__)
        self._load_bloodline_master()
    
    def _load_bloodline_master(self) -> None:
        """血統マスタファイルを読み込み、辞書形式で保存"""
        try:
            df = pd.read_csv(self.master_file_path, encoding='utf-8-sig')
            
            # 必要な列があるかチェック
            required_cols = ['馬名', '小系統', '国系統']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"血統マスタに必要な列がありません: {missing_cols}")
            
            # 血統辞書を構築
            for _, row in df.iterrows():
                horse_name = self.normalize_text(row['馬名'])
                if horse_name:
                    self.bloodline_dict[horse_name] = (
                        row['小系統'] if pd.notna(row['小系統']) else 'UNKNOWN',
                        row['国系統'] if pd.notna(row['国系統']) else 'UNKNOWN'
                    )
            
            self.logger.info(f"血統マスタを読み込みました: {len(self.bloodline_dict)}頭の馬")
            
        except Exception as e:
            self.logger.error(f"血統マスタの読み込みに失敗: {e}")
            raise RuntimeError(f"血統マスタ読み込みエラー: {e}") from e
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """テキストの正規化"""
        if pd.isna(text):
            return ''
        return unicodedata.normalize('NFKC', str(text)).strip()
    
    def lookup_bloodline(self, horse_name: str) -> Tuple[str, str]:
        """馬名から血統情報を検索"""
        normalized_name = self.normalize_text(horse_name)
        
        if normalized_name in self.bloodline_dict:
            return self.bloodline_dict[normalized_name]
        else:
            self.logger.warning(f"血統情報が見つかりません: {horse_name}")
            return ('UNKNOWN', 'UNKNOWN')
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameに血統情報を追加"""
        result_df = df.copy()
        
        # デバッグ情報
        found_count = 0
        not_found_horses = []
        
        # 父馬名と母の父馬名の血統情報を追加
        for horse_type, prefix in [('父馬名', '父'), ('母の父馬名', '母父')]:
            if horse_type not in result_df.columns:
                self.logger.warning(f"列 '{horse_type}' が見つかりません")
                continue
            
            small_lineages = []
            country_lineages = []
            
            for horse_name in result_df[horse_type]:
                small, country = self.lookup_bloodline(horse_name)
                small_lineages.append(small)
                country_lineages.append(country)
                
                # デバッグ: 見つかったかカウント
                if small != 'UNKNOWN' and country != 'UNKNOWN':
                    found_count += 1
                else:
                    not_found_horses.append(horse_name)
            
            result_df[f'{prefix}_小系統'] = small_lineages
            result_df[f'{prefix}_国系統'] = country_lineages
        
        self.logger.info(f"血統情報を追加: 成功={found_count}件, 未発見={len(not_found_horses)}件")
        
        # 未発見の馬をログ出力
        if not_found_horses:
            self.logger.warning(f"血統マスタに見つからなかった馬: {set(not_found_horses)}")
        
        return result_df, not_found_horses  # 未発見リストも返す

class KeibaV4PredictionApp:
    """V4競馬予測システム メインアプリケーション"""
    
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_columns = None
        self.model_loaded = False
        self.bloodline_manager = None
        
        # 血統マネージャー読み込み
        self.load_bloodline_manager()
        
        # モデル読み込み
        self.load_v4_model()
    
    def load_bloodline_manager(self):
        """血統マスタを読み込み"""
        try:
            bloodline_paths = [
                Path("data/bloodline_master.csv"),
                Path("./data/bloodline_master.csv"),
                Path("../data/bloodline_master.csv")
            ]
            
            for bloodline_path in bloodline_paths:
                if bloodline_path.exists():
                    self.bloodline_manager = BloodlineManager(str(bloodline_path))
                    logger.info(f"血統マスタ読み込み成功: {bloodline_path}")
                    return
            
            logger.warning("⚠️ 血統マスタファイルが見つかりません。血統補完機能は無効です。")
            
        except Exception as e:
            logger.error(f"血統マスタ読み込みエラー: {str(e)}")
            st.warning("⚠️ 血統マスタの読み込みに失敗しました。血統補完機能は無効です。")
    
    def load_v4_model(self):
        """V4モデルとエンコーダーを読み込み"""
        try:
            # GitHub Codespaces/Streamlit Cloud用のパス調整
            model_paths = [
                # ローカル開発用
                Path("ml_models_v4/models/lgb_v4_time_model_20251108_211745.pkl"),
                Path("../ml_models_v4/models/lgb_v4_time_model_20251108_211745.pkl"),
                # デプロイ用
                Path("models/lgb_v4_time_model.pkl"),
                Path("./models/lgb_v4_time_model.pkl")
            ]
            
            encoder_paths = [
                Path("ml_models_v4/models/label_encoders_v4_20251108_211745.pkl"),
                Path("../ml_models_v4/models/label_encoders_v4_20251108_211745.pkl"),
                Path("models/label_encoders_v4.pkl"),
                Path("./models/label_encoders_v4.pkl")
            ]
            
            feature_paths = [
                Path("ml_models_v4/models/feature_columns_v4_20251108_211745.json"),
                Path("../ml_models_v4/models/feature_columns_v4_20251108_211745.json"),
                Path("models/feature_columns_v4.json"),
                Path("./models/feature_columns_v4.json")
            ]
            
            # モデル読み込み
            model_loaded = False
            for model_path in model_paths:
                if model_path.exists():
                    self.model = joblib.load(model_path)
                    model_loaded = True
                    logger.info(f"モデル読み込み成功: {model_path}")
                    break
            
            if not model_loaded:
                st.error("❌ V4モデルファイルが見つかりません")
                return
            
            # エンコーダー読み込み
            encoder_loaded = False
            for encoder_path in encoder_paths:
                if encoder_path.exists():
                    self.encoders = joblib.load(encoder_path)
                    encoder_loaded = True
                    logger.info(f"エンコーダー読み込み成功: {encoder_path}")
                    break
            
            if not encoder_loaded:
                st.error("❌ エンコーダーファイルが見つかりません")
                return
            
            # 特徴量リスト読み込み
            features_loaded = False
            for feature_path in feature_paths:
                if feature_path.exists():
                    with open(feature_path, 'r', encoding='utf-8') as f:
                        self.feature_columns = json.load(f)
                    features_loaded = True
                    logger.info(f"特徴量リスト読み込み成功: {feature_path}")
                    break
            
            if not features_loaded:
                st.error("❌ 特徴量ファイルが見つかりません")
                return
            
            self.model_loaded = True
            st.success("✅ V4モデル読み込み完了")
            
        except Exception as e:
            st.error(f"❌ モデル読み込みエラー: {str(e)}")
            logger.error(f"モデル読み込みエラー: {traceback.format_exc()}")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """V4モデル用データ前処理"""
        try:
            df = df.copy()
            
            # 0. 血統情報の欠損値をUNKNOWNに置き換え
            bloodline_cols = ['父_小系統', '父_国系統', '母父_小系統', '母父_国系統']
            for col in bloodline_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('UNKNOWN').replace('', 'UNKNOWN')
            
            # 1. カテゴリカル変数のエンコーディング
            categorical_cols = [
                '場所', '芝・ダ', '馬場状態', '性別', '騎手名', '調教師',
                '父_小系統', '父_国系統', '母父_小系統', '母父_国系統'
            ]
            
            for col in categorical_cols:
                if col in df.columns and col in self.encoders:
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.encoders[col].transform([x])[0] 
                        if x in self.encoders[col].classes_ else 0
                    )
            
            # 2. 数値特徴量の変換
            if '単勝オッズ' in df.columns:
                df['単勝オッズ_log'] = np.log1p(df['単勝オッズ'].fillna(df['単勝オッズ'].median()))
            
            # 3. 日付特徴量
            if all(col in df.columns for col in ['年', '月']):
                df['年月'] = df['年'] * 100 + df['月']
                df['季節'] = df['月'].apply(self._get_season)
            
            # 4. 組み合わせ特徴量
            if all(col in df.columns for col in ['距離', '芝・ダ']):
                df['距離_表面'] = df['距離'].astype(str) + '_' + df['芝・ダ'].astype(str)
                if '距離_表面' in self.encoders:
                    df['距離_表面_encoded'] = df['距離_表面'].apply(
                        lambda x: self.encoders['距離_表面'].transform([x])[0] 
                        if x in self.encoders['距離_表面'].classes_ else 0
                    )
            
            # 5. 血統組み合わせ
            if all(col in df.columns for col in ['父_小系統', '母父_小系統']):
                df['血統組み合わせ'] = df['父_小系統'].astype(str) + '_' + df['母父_小系統'].astype(str)
                if '血統組み合わせ' in self.encoders:
                    df['血統組み合わせ_encoded'] = df['血統組み合わせ'].apply(
                        lambda x: self.encoders['血統組み合わせ'].transform([x])[0] 
                        if x in self.encoders['血統組み合わせ'].classes_ else 0
                    )
            
            return df
            
        except Exception as e:
            st.error(f"❌ データ前処理エラー: {str(e)}")
            logger.error(f"前処理エラー: {traceback.format_exc()}")
            return df
    
    def _get_season(self, month: int) -> int:
        """月から季節を取得"""
        if month in [12, 1, 2]:
            return 0  # 冬
        elif month in [3, 4, 5]:
            return 1  # 春
        elif month in [6, 7, 8]:
            return 2  # 夏
        else:
            return 3  # 秋
    
    def predict_race_time(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """レースタイム予測"""
        if not self.model_loaded:
            st.error("❌ モデルが読み込まれていません")
            return None
        
        try:
            # データ前処理
            processed_df = self.preprocess_data(df)
            
            # 必要な特徴量の確認
            available_features = [col for col in self.feature_columns if col in processed_df.columns]
            missing_features = [col for col in self.feature_columns if col not in processed_df.columns]
            
            if missing_features:
                st.warning(f"⚠️ 不足特徴量: {missing_features}")
            
            # 予測実行
            X = processed_df[available_features].copy()
            X = X.fillna(X.median())  # 欠損値処理
            
            predicted_times = self.model.predict(X)
            
            # 結果をDataFrameに追加
            result_df = df.copy()
            result_df['予測タイム'] = predicted_times
            result_df['予測順位'] = result_df['予測タイム'].rank(method='min')
            
            return result_df
            
        except Exception as e:
            st.error(f"❌ 予測エラー: {str(e)}")
            logger.error(f"予測エラー: {traceback.format_exc()}")
            return None
    
    def run(self):
        """メインアプリケーション実行"""
        # ヘッダー
        st.title("🏇 競馬予測システム V4")
        st.markdown("### レース前情報による高精度タイム予測")
        
        if not self.model_loaded:
            st.error("❌ モデルの読み込みに失敗しました。管理者にお問い合わせください。")
            return
        
        # サイドバー情報
        st.sidebar.markdown("## 📊 V4システム情報")
        st.sidebar.markdown("""
        **予測精度（2025年実測）:**
        - MAE: 0.961秒
        - RMSE: 1.841秒  
        - 精度: 90.6% (2秒以内)
        
        **使用特徴量:**
        - 基本レース条件
        - 馬・騎手・調教師情報
        - 血統系統情報
        - 人気・オッズ情報
        
        **特徴:**
        - ✅ レース前情報のみ使用
        - ✅ 実用的な予測精度
        - ✅ タイム→順位変換
        """)
        
        # メインコンテンツ
        tab1, tab2, tab3 = st.tabs(["📁 CSV一括予測", "✍️ 手動入力", "📚 使い方"])
        
        with tab1:
            self.csv_prediction_interface()
        
        with tab2:
            self.manual_input_interface()
        
        with tab3:
            self.usage_guide()
    
    def csv_prediction_interface(self):
        """CSV一括予測インターフェース"""
        st.markdown("## 📁 CSV一括予測")
        st.markdown("レースデータをCSVファイルでアップロードして一括予測を実行します。")
        
        # CSVフォーマット説明
        with st.expander("📋 必要なCSVフォーマット"):
            st.markdown("""
            ### 必須カラム:
            - **基本情報**: `距離`, `頭数`, `馬番`, `年齢`, `斤量`
            - **レース条件**: `場所`, `芝・ダ`, `馬場状態`
            - **人間情報**: `騎手名`, `調教師`, `性別`
            - **人気情報**: `人気順`, `単勝オッズ`
            - **日付情報**: `年`, `月`, `日`
            - **識別情報**: `馬名`
            
            ### 血統情報（以下のいずれか）:
            
            **パターン1: 血統系統を直接指定**
            - `父_小系統`, `父_国系統`, `母父_小系統`, `母父_国系統`
            
            **パターン2: 馬名から自動補完（推奨）**
            - `父馬名`, `母の父馬名` → システムが自動的に系統情報を補完します
            
            ### サンプルデータ（パターン1）:
            ```csv
            馬名,年,月,日,場所,芝・ダ,距離,馬場状態,馬番,性別,年齢,騎手名,調教師,斤量,頭数,人気順,単勝オッズ,父_小系統,父_国系統,母父_小系統,母父_国系統
            サンプル馬,25,11,10,東京,芝,2000,良,1,牡,4,騎手A,調教師B,57,16,1,2.1,ディープ系,日本型サンデー系,キングマンボ系,欧州型ミスプロ系
            ```
            
            ### サンプルデータ（パターン2 - 自動補完）:
            ```csv
            馬名,年,月,日,場所,芝・ダ,距離,馬場状態,馬番,性別,年齢,騎手名,調教師,斤量,頭数,人気順,単勝オッズ,父馬名,母の父馬名
            サンプル馬,25,11,10,東京,芝,2000,良,1,牡,4,騎手A,調教師B,57,16,1,2.1,ディープインパクト,キングマンボ
            ```
            
            ※ パターン2の場合、システムが自動的に血統系統情報を付与します
            """)
        
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            "CSVファイルを選択してください",
            type=['csv'],
            help="上記フォーマットに従ったCSVファイルをアップロードしてください"
        )
        
        if uploaded_file is not None:
            try:
                # CSV読み込み
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                st.success(f"✅ ファイル読み込み完了: {len(df)}件")
                
                # データプレビュー
                st.markdown("### 📊 アップロードデータプレビュー")
                st.dataframe(df.head(10), use_container_width=True)
                
                # 血統情報の自動補完チェック
                bloodline_cols = ['父_小系統', '父_国系統', '母父_小系統', '母父_国系統']
                missing_bloodline = [col for col in bloodline_cols if col not in df.columns]
                
                if missing_bloodline and self.bloodline_manager:
                    # 父馬名と母の父馬名があるかチェック
                    if '父馬名' in df.columns and '母の父馬名' in df.columns:
                        st.info("🧬 血統情報が不足しています。父馬名・母の父馬名から自動補完します...")
                        
                        with st.spinner("血統情報を補完中..."):
                            df, not_found_horses = self.bloodline_manager.enrich_dataframe(df)
                        
                        st.success("✅ 血統情報を補完しました")
                        
                        # 補完統計情報
                        total_horses = len(df) * 2  # 父 + 母父
                        found_count = total_horses - len(not_found_horses)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("総馬数", f"{total_horses}頭")
                        with col2:
                            st.metric("補完成功", f"{found_count}頭", delta=f"{found_count/total_horses*100:.1f}%")
                        with col3:
                            st.metric("未発見", f"{len(not_found_horses)}頭")
                        
                        # 未発見の馬リスト表示
                        if not_found_horses:
                            with st.expander("⚠️ 血統マスタに見つからなかった馬（UNKNOWN設定）"):
                                unique_not_found = sorted(set(not_found_horses))
                                st.write(", ".join(unique_not_found))
                        
                        # 補完結果のサンプル表示
                        st.markdown("#### 📋 補完後データサンプル（先頭5行）")
                        sample_cols = ['馬名', '父馬名', '父_小系統', '父_国系統', '母の父馬名', '母父_小系統', '母父_国系統']
                        st.dataframe(df[sample_cols].head(5), use_container_width=True)
                        
                        # 補完後の全データ表示
                        with st.expander("📊 補完後の全データを表示"):
                            st.dataframe(df, use_container_width=True, height=400)
                    else:
                        st.error("❌ 血統情報の補完には `父馬名` と `母の父馬名` の列が必要です")
                        st.stop()
                
                # 必要カラムチェック
                required_cols = [
                    '馬名', '年', '月', '日', '場所', '芝・ダ', '距離', '馬場状態',
                    '馬番', '性別', '年齢', '騎手名', '調教師', '斤量', '頭数',
                    '人気順', '単勝オッズ', '父_小系統', '父_国系統', '母父_小系統', '母父_国系統'
                ]
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ 不足カラム: {missing_cols}")
                    st.stop()
                
                # 予測実行
                if st.button("🚀 予測実行", type="primary"):
                    with st.spinner("予測中..."):
                        result_df = self.predict_race_time(df)
                    
                    if result_df is not None:
                        st.markdown("### 🎯 予測結果")
                        
                        # 結果表示
                        display_cols = ['馬名', '予測タイム', '予測順位', '人気順', '単勝オッズ']
                        st.dataframe(
                            result_df[display_cols].sort_values('予測順位'),
                            use_container_width=True
                        )
                        
                        # 統計情報
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("最速予想", f"{result_df['予測タイム'].min():.1f}秒")
                        with col2:
                            st.metric("最遅予想", f"{result_df['予測タイム'].max():.1f}秒")
                        with col3:
                            st.metric("タイム幅", f"{result_df['予測タイム'].max() - result_df['予測タイム'].min():.1f}秒")
                        
                        # CSV下载
                        csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            "📥 結果をCSVダウンロード",
                            data=csv,
                            file_name=f"keiba_prediction_v4_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ ファイル処理エラー: {str(e)}")
    
    def manual_input_interface(self):
        """手動入力インターフェース"""
        st.markdown("## ✍️ 手動入力予測")
        st.markdown("1頭ずつ詳細に入力して予測を実行します。")
        
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 基本情報")
                horse_name = st.text_input("馬名", value="サンプル馬")
                year = st.number_input("年", min_value=20, max_value=30, value=25)
                month = st.number_input("月", min_value=1, max_value=12, value=11)
                day = st.number_input("日", min_value=1, max_value=31, value=10)
                
                st.markdown("### レース条件")
                location = st.selectbox("競馬場", ["東京", "中山", "阪神", "京都", "新潟", "小倉", "函館", "札幌", "中京", "福島"])
                surface = st.selectbox("芝・ダート", ["芝", "ダ"])
                distance = st.number_input("距離(m)", min_value=1000, max_value=4000, value=2000, step=100)
                track_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])
                
            with col2:
                st.markdown("### 馬情報")
                horse_number = st.number_input("馬番", min_value=1, max_value=18, value=1)
                gender = st.selectbox("性別", ["牡", "牝", "セ"])
                age = st.number_input("年齢", min_value=2, max_value=10, value=4)
                weight = st.number_input("斤量", min_value=48.0, max_value=65.0, value=57.0, step=0.5)
                field_size = st.number_input("頭数", min_value=5, max_value=18, value=16)
                
                st.markdown("### 人気・オッズ")
                popularity = st.number_input("人気順", min_value=1, max_value=18, value=1)
                odds = st.number_input("単勝オッズ", min_value=1.0, max_value=999.9, value=2.1, step=0.1)
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("### 人的要因")
                jockey = st.text_input("騎手名", value="騎手A")
                trainer = st.text_input("調教師", value="調教師B")
                
            with col4:
                st.markdown("### 血統情報")
                father_small = st.selectbox("父_小系統", [
                    "ディープ系", "キングマンボ系", "Tサンデー系", "ロベルト系", 
                    "Pサンデー系", "ストームバード系", "ミスプロ系", "その他"
                ])
                father_large = st.selectbox("父_国系統", [
                    "日本型サンデー系", "欧州型ミスプロ系", "米国型ノーザンダンサー系",
                    "欧州型ノーザンダンサー系", "米国型ミスプロ系", "その他"
                ])
                mother_small = st.selectbox("母父_小系統", [
                    "キングマンボ系", "ディープ系", "Tサンデー系", "ミスプロ系",
                    "Pサンデー系", "ロベルト系", "その他"
                ])
                mother_large = st.selectbox("母父_国系統", [
                    "欧州型ミスプロ系", "日本型サンデー系", "米国型ノーザンダンサー系",
                    "欧州型ノーザンダンサー系", "米国型ミスプロ系", "その他"
                ])
            
            submitted = st.form_submit_button("🎯 予測実行", type="primary")
            
            if submitted:
                # 入力データをDataFrameに変換
                input_data = {
                    '馬名': [horse_name],
                    '年': [year], '月': [month], '日': [day],
                    '場所': [location], '芝・ダ': [surface], '距離': [distance], '馬場状態': [track_condition],
                    '馬番': [horse_number], '性別': [gender], '年齢': [age], '斤量': [weight], '頭数': [field_size],
                    '騎手名': [jockey], '調教師': [trainer],
                    '人気順': [popularity], '単勝オッズ': [odds],
                    '父_小系統': [father_small], '父_国系統': [father_large],
                    '母父_小系統': [mother_small], '母父_国系統': [mother_large]
                }
                
                df = pd.DataFrame(input_data)
                
                with st.spinner("予測中..."):
                    result_df = self.predict_race_time(df)
                
                if result_df is not None:
                    predicted_time = result_df['予測タイム'].iloc[0]
                    
                    st.markdown("### 🎯 予測結果")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("予測タイム", f"{predicted_time:.2f}秒")
                    with col2:
                        st.metric("距離", f"{distance}m")
                    with col3:
                        st.metric("ペース", f"{predicted_time/distance*1000:.1f}秒/km")
                    
                    # 詳細情報
                    st.markdown("### 📊 詳細情報")
                    info_df = pd.DataFrame({
                        '項目': ['馬名', '競馬場', '距離', '馬場', '騎手', '人気', 'オッズ', '予測タイム'],
                        '値': [horse_name, location, f"{distance}m", track_condition, 
                              jockey, f"{popularity}番人気", f"{odds}倍", f"{predicted_time:.2f}秒"]
                    })
                    st.dataframe(info_df, use_container_width=True)
    
    def usage_guide(self):
        """使い方ガイド"""
        st.markdown("## 📚 競馬予測システム V4 使い方ガイド")
        
        st.markdown("""
        ### 🎯 V4システムの特徴
        
        **高精度予測**:
        - 2025年実測で平均誤差0.961秒を達成
        - 90.6%の馬が2秒以内の精度で予測
        - プロ予想家レベルの順位予測精度
        
        **実用性**:
        - レース前情報のみ使用で実際に予測可能
        - タイム予測→順位変換で安定した結果
        - 血統・騎手・人気情報を総合的に評価
        
        ### 📁 CSV一括予測の使い方
        
        1. **フォーマット準備**: 必要な21項目を含むCSVを準備
        2. **ファイルアップロード**: CSVファイルをドラッグ&ドロップ
        3. **データ確認**: アップロードされたデータをプレビュー
        4. **予測実行**: ボタンクリックで全頭一括予測
        5. **結果ダウンロード**: 予測結果をCSVで取得
        
        ### ✍️ 手動入力の使い方
        
        1. **基本情報**: 馬名、日付、レース条件を入力
        2. **馬情報**: 馬番、年齢、斤量などの詳細
        3. **人的要因**: 騎手、調教師情報
        4. **血統情報**: 父・母父の系統分類
        5. **予測実行**: 即座にタイム予測結果を表示
        
        ### 🎲 予測結果の活用法
        
        **単勝戦略**:
        - 予測1位の馬への投資
        - 人気薄で予測上位の馬を狙い撃ち
        
        **複勝戦略**:
        - 予測Top3への分散投資
        - 高い的中率で安定収益
        
        **穴馬発見**:
        - 人気順 vs 予測順位の乖離をチェック
        - 人気薄×予測上位 = 高配当候補
        
        ### ⚠️ 注意事項
        
        - 予測は統計的手法に基づく推定値です
        - 競馬には不確定要素が多く含まれます  
        - 投資は自己責任で行ってください
        - システムの結果を過信せず、総合的に判断してください
        
        ### 📞 サポート情報
        
        - GitHub: https://github.com/Yu10Kumura/keiba-prediction-app
        - モデルバージョン: V4 (2025年11月版)
        - 最終更新: 2025年11月8日
        """)

def main():
    """メイン実行関数"""
    try:
        app = KeibaV4PredictionApp()
        app.run()
    except Exception as e:
        st.error(f"❌ アプリケーションエラー: {str(e)}")
        logger.error(f"アプリケーションエラー: {traceback.format_exc()}")

if __name__ == "__main__":
    main()