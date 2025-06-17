from typing import Tuple

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, PreTrainedTokenizer, AutoConfig
#AutoProcessor: マルチモーダルモデル向けに画像＋テキストを一括で前処理するユーティリティ。
#LlavaOnevisionForConditionalGeneration: LLaVA‐OneVision のモデル本体クラス。Hugging Face Transformers で提供されています。
#PreTrainedTokenizer: トークナイザーの基底クラス。ここではプロセッサから取り出して型を保証します。
#AutoConfig: モデル設定オブジェクトを読み込むためのファクトリクラス。

from . import register_loader #loaders/__init__.py で定義されたデコレータ。
from .base import BaseModelLoader


###LLaVA‐OneVision モデルとそれに必要なトークナイザー・プロセッサ・設定をまとめて読み込むための “ローダー” クラスを定義###

@register_loader("llava-onevision")  #これをつけると、このクラスが LOADERS 辞書に "llava-onevision" というキーで自動登録されます。
class LLaVAOnevisionModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaOnevisionForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]: #Tupleは戻り値のヒント
        if load_model:
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs, #**:辞書をキーワード引数として展開する
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config

#load() メソッドで、
# モデル本体（オプション）
# プロセッサ（画像＋テキスト前処理のユーティリティ）
# トークナイザー（テキストのトークン化用）
# 設定オブジェクト（アーキテクチャ情報）
# を返し、ファインチューニングパイプラインの準備を完了させます。

#この構造により、異なるモデルファミリを扱いたいときは、同様の形式で新しいローダーを loaders/ に追加するだけで、train.py を変更せずに済むようになっています。
