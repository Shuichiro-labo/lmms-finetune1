from typing import Dict, Optional, List
from dataclasses import dataclass, field

import transformers

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES

###Hugging Face の HfArgumentParser と連携して、コマンドライン引数を型付きで受け取りやすくするためのもの###
#arguments.py は、train.py がコマンドラインから受け取りたい引数を 型付きの dataclass でまとめたファイルです。

#モデルを指定するクラス
#基本いじらなくていいはず
@dataclass
class ModelArguments:
    model_id: str = field(default="llava-1.5-7b") #model_idの設定 defaultはllava-1.5-7bになっている
    model_local_path: Optional[str] = field(default=None) #ローカルにキャッシュ済みや手元にダウンロード済みのモデルディレクトリを明示したい場合に指定。
                                                          #指定しないと model_hf_path（Hub のパス）をそのまま使います。

    #supported_models.pyの値が関係してくる
    def __post_init__(self):
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_hf_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]

        if not self.model_local_path:
            self.model_local_path = self.model_hf_path

#データセット関連を指定するクラス
#基本いじらなくていい
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data json file."}
    ) #学習用 JSON データセットのパス。
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data json file."}
    ) #検証用 JSON を別途指定する場合 未指定なら評価なしモードになる
    #JSON 内で指定した相対パスの起点となるディレクトリ。画像・動画混合のサンプルを扱うときに使う。
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8) #動画から抽出するフレーム数。
    #JSON 中の会話履歴で、「ユーザー側」を示すキー名／「アシスタント側」を示すキー名をそれぞれ指定。
    user_key: Optional[str] = field(default="human") #変えるとしてもここの名前ぐらい
    assistant_key: Optional[str] = field(default="gpt")


#Hugging Face Trainerが理解する基本的な学習オプションをすべて引き継ぐ
#バッチサイズ、エポック数、学習率、DeepSpeed 設定、出力先ディレクトリ、ログ頻度など、Hugging Face Trainer が理解する基本的な学習オプションをすべて引き継ぎます
#出力のtraining.ymlにはここには書いていない親クラスにある引数も書かれる
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = field(default=False) #flashattentionを使うか
    train_vision_encoder: bool = field(default=False)
    train_vision_projector: bool = field(default=False)
    mask_question_tokens: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False #remove_unused_columns は Hugging Face の Trainer クラスが持っているオプションで、
                                           # デフォルトでは True に設定されており、
                                           # データセット（Dataset）から「モデルの forward() に使われないカラム」を自動で取り除く動作をします。
                                           #マルチモーダル（画像や動画、対話履歴などを混ぜた）データの場合、Trainer はあくまでテキスト用に設計されているため、JSON に含まれる image や video、conversations といった「テキスト以外のカラム」を「unused」と見なして削除しようとしてしまいます。
                                           # remove_unused_columns=False をセットすることで、テキスト以外のカラム（画像や動画情報など）を削除されずに Trainer がバッチを組めるようにしている。

###LoRA関連のオプション###
@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    use_vision_lora: bool = field(default=True)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"
