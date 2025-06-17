from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional

import torch
from transformers import PreTrainedTokenizer, AutoProcessor, AutoConfig #PreTrainedTokenizer：トークナイザーの共通親クラス
                                                                        #AutoProcessor：画像＋テキストなど複合データ用の前処理器
                                                                        #AutoConfig：モデル設定を読み込むクラス

#Python の組み込みモジュール abc（Abstract Base Classes）から。
#ABC を継承することで「抽象クラス」を定義
#@abstractmethod デコレータを付けたメソッドはサブクラスで必ずオーバーライドが必要

#「教師ありファインチューニング用にサンプルをまとめる（collate）」ことを想定した基底クラス
class BaseDataCollator(ABC, object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self,
        config: Optional[AutoConfig] = None, #モデル設定オブジェクト（例えば image_token_index などを持つ）
        tokenizer: Optional[PreTrainedTokenizer] = None, #テキストトークナイザー
        processor: Optional[AutoProcessor] = None, #画像＋テキスト複合データを前処理するオブジェクト
        mask_question_tokens: bool = True #画像＋テキスト複合データを前処理するオブジェクト
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
    

    #PyTorch の損失関数（CrossEntropyLoss など）で「ラベル -100 は無視する」という慣習を利用。
    #具体的には、質問トークンやパディングトークンを損失計算から除外するときにこの値を使います。
    @property
    def IGNORE_TOKEN_ID(self) -> int:
        return -100

    #input_ids のパディングに用いる特殊トークンID。それを使って attention mask を作ったり、シーケンス長を揃えたりします。
    #トークナイザーに pad_token_id が設定されていないとエラーになるので、tokenizer は必ず pad_token を持つモデルである必要があります。
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id


    #Sequence[Dict]：__call__ に渡される「複数のサンプル」を表す
    #引数：instances は「１バッチ分のサンプル」をまとめた Sequence[Dict]
    #戻り値：Dict[str, torch.Tensor] 典型的には {"input_ids": Tensor, "labels": Tensor, "attention_mask": Tensor, …} に加えて、画像や動画の pixel_values などを含む辞書
    #「サブクラスで必ず実装すべき」メソッドで、渡された生データ（instances）を Transformer モデルに与えられるテンソル群に変換する処理を書きます。
    @abstractmethod
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]: ...
