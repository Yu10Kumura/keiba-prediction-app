# 血統マスタ管理機能
"""
血統マスタファイルから馬名を検索し、小系統・国系統を取得する機能
添付スクリプトのロジックをクラス化して再利用可能にしたもの
"""

import pandas as pd
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from ..constants.columns import BLOODLINE_MASTER_COLUMNS, DEFAULT_VALUES
from ..constants.messages import ERROR_MESSAGES, WARNING_MESSAGES

class BloodlineManager:
    """血統マスタ管理クラス"""
    
    def __init__(self, master_file_path: str):
        """
        血統マスタファイルを読み込んで初期化
        
        Args:
            master_file_path: 血統マスタCSVファイルのパス
        """
        self.master_file_path = Path(master_file_path)
        self.bloodline_dict = {}
        self.logger = logging.getLogger(__name__)
        self._load_bloodline_master()
    
    def _load_bloodline_master(self) -> None:
        """血統マスタファイルを読み込み、辞書形式で保存"""
        try:
            # CSVファイル読み込み
            df = pd.read_csv(
                self.master_file_path, 
                encoding='utf-8-sig'
            )
            
            # 必要な列があるかチェック
            horse_name_col = BLOODLINE_MASTER_COLUMNS['horse_name']
            small_lineage_col = BLOODLINE_MASTER_COLUMNS['small_lineage']
            country_lineage_col = BLOODLINE_MASTER_COLUMNS['country_lineage']
            
            required_cols = [horse_name_col, small_lineage_col, country_lineage_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"血統マスタに必要な列がありません: {missing_cols}")
            
            # 血統辞書を構築
            self.bloodline_dict = {}
            for _, row in df.iterrows():
                horse_name = self.normalize_text(row[horse_name_col])
                if horse_name:  # 空でない場合のみ追加
                    self.bloodline_dict[horse_name] = (
                        row[small_lineage_col] if pd.notna(row[small_lineage_col]) else DEFAULT_VALUES['unknown_bloodline'],
                        row[country_lineage_col] if pd.notna(row[country_lineage_col]) else DEFAULT_VALUES['unknown_bloodline']
                    )
            
            self.logger.info(f"血統マスタを読み込みました: {len(self.bloodline_dict)}頭の馬")
            
        except Exception as e:
            self.logger.error(f"血統マスタの読み込みに失敗: {e}")
            raise RuntimeError(ERROR_MESSAGES['bloodline_lookup_error']) from e
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        テキストの正規化（添付スクリプトと同じロジック）
        
        Args:
            text: 正規化対象のテキスト
            
        Returns:
            正規化されたテキスト
        """
        if pd.isna(text):
            return DEFAULT_VALUES['empty_string']
        
        # NFKC正規化 + 前後空白除去
        return unicodedata.normalize('NFKC', str(text)).strip()
    
    def lookup_bloodline(self, horse_name: str) -> Tuple[str, str]:
        """
        馬名から血統情報を検索
        
        Args:
            horse_name: 検索対象の馬名
            
        Returns:
            (小系統, 国系統) のタプル。見つからない場合は ('UNK', 'UNK')
        """
        normalized_name = self.normalize_text(horse_name)
        
        if normalized_name in self.bloodline_dict:
            return self.bloodline_dict[normalized_name]
        else:
            # 見つからない場合は警告ログ出力
            self.logger.warning(f"血統情報が見つかりません: {horse_name}")
            return (
                DEFAULT_VALUES['unknown_bloodline'], 
                DEFAULT_VALUES['unknown_bloodline']
            )
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameに血統情報を追加（添付スクリプトと同じ処理）
        
        Args:
            df: 血統情報を追加するDataFrame
            
        Returns:
            血統情報が追加されたDataFrame
        """
        result_df = df.copy()
        
        # 父馬名と母の父馬名の血統情報を追加
        for horse_type in ['父馬名', '母の父馬名']:
            if horse_type not in result_df.columns:
                self.logger.warning(f"列 '{horse_type}' が見つかりません")
                continue
            
            # 各行に対して血統情報を取得
            for idx, row in result_df.iterrows():
                horse_name = row[horse_type]
                small_lineage, country_lineage = self.lookup_bloodline(horse_name)
                
                # 血統情報を設定
                result_df.at[idx, f'{horse_type}_小系統'] = small_lineage
                result_df.at[idx, f'{horse_type}_国系統'] = country_lineage
        
        self.logger.info(f"血統情報を追加しました: {len(result_df)}行")
        return result_df
    
    def get_bloodline_stats(self) -> Dict[str, int]:
        """血統マスタの統計情報を取得"""
        return {
            'total_horses': len(self.bloodline_dict),
            'unique_small_lineages': len(set(lineage[0] for lineage in self.bloodline_dict.values())),
            'unique_country_lineages': len(set(lineage[1] for lineage in self.bloodline_dict.values()))
        }
    
    def search_horses_by_lineage(self, small_lineage: str = None, country_lineage: str = None) -> list:
        """
        系統で馬を検索
        
        Args:
            small_lineage: 小系統名
            country_lineage: 国系統名
            
        Returns:
            該当する馬名のリスト
        """
        matches = []
        
        for horse_name, (s_lineage, c_lineage) in self.bloodline_dict.items():
            if small_lineage and s_lineage != small_lineage:
                continue
            if country_lineage and c_lineage != country_lineage:
                continue
            matches.append(horse_name)
        
        return sorted(matches)
    
    def reload_master(self) -> None:
        """血統マスタを再読み込み"""
        self.logger.info("血統マスタを再読み込みします")
        self._load_bloodline_master()


# ユーティリティ関数
def create_bloodline_manager(config: dict) -> BloodlineManager:
    """
    設定から血統マネージャーを作成
    
    Args:
        config: アプリケーション設定辞書
        
    Returns:
        BloodlineManagerインスタンス
    """
    bloodline_path = config['paths']['bloodline_master']
    return BloodlineManager(bloodline_path)
