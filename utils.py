import math
from typing import List, Dict, Optional

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import transformers
from transformers import Trainer
from transformers.trainer import has_length



#役割：DeepSpeed を用いた分散学習時に「テキストのみのサンプル」だけでバッチが構成されないようにサンプリングを調整する。
#背景：DeepSpeed の ZeRO Offload では、すべての GPU プロセスにマルチモーダル（画像＋テキストなど）が必ず含まれている必要があるケースがあるため。
class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int, #各GPUあたりのミニバッチサイズ
        world_size: int, #GPU数 × 勾配蓄積ステップ数（“mega batch” のサイズ計算に使う）
        is_text_only: Optional[List[bool]] = None, #データセット各サンプルが“テキストのみ”かどうかを示すフラグリスト
        generator=None, #乱数ジェネレータ（オプション）
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size #全体のバッチサイズ
                                                       #1台のGPUで扱える実質バッチサイズ（各GPUあたりのミニバッチサイズ×勾配蓄積ステップ数）× GPU数　この考え方の方がイメージしやすい

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only] #マルチモーダル（テキスト＋画像／動画）サンプルのインデックス
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only] #テキストのみサンプルのインデックス

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size) #全サンプル数 ÷ mega_batch_size を切り上げて num_batches を決定
        if len(mm_indices) < num_batches: #もしマルチモーダル数がバッチ数より少なければエラーを投げる
            raise ValueError(
                f"{len(mm_indices)} multimodal entries, {len(num_batches)} batches. "
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed. "
                "Try increasing the batch size first."
            )

        # shuffle indices
        #torch.randperm で mm_indicesとuni_indices をランダム順に並べ替え
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        #メガバッチ（全プロセス分）ごとの組み立て、各メガバッチに割り当てるテキストのみ数を均等配分し、残りをマルチモーダルサンプルで埋める。最後のバッチだけは残ったマルチモーダルサンプルを全部追加
        #テキストのみサンプル数の均等分配
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        #メガバッチの組み立てループ
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []

            # 2-1. テキストのみサンプルを追加
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            # 2-2. マルチモーダルサンプルを追加
            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                # 最終バッチ：残りの mm_indices をすべて追加
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        #メガバッチ間のシャッフル　データセット先頭の偏りを防ぎ、学習を安定化させる
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        #平坦化して最終的なインデックス列を生成
        #各メガバッチ（リスト）の中身を１つの大きなリストに連結
        #DataLoader はこの順序でサンプルを取り出し、かつ内部でバッチサイズごとに区切ることで、「メガバッチ」 → 「各プロセスが受け取る小バッチ」という形で分散学習に渡されます。
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


#TrainerWithCustomSampler クラスは、HuggingFace の標準 Trainer を継承し、
#学習時 (_get_train_sampler)
#評価時 (_get_eval_sampler)
# ――それぞれで使うサンプラーを、デフォルトの DistributedSampler → NoTextOnlyBatchSampler に置き換えることで、
# 「テキストのみサンプルだけのミニバッチができないようにする」
# DeepSpeed（ZeRO Stage 3） アプライアンスやマルチモーダル学習で必須となる制約を満たす
class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # データがない or 長さ不明なら通常の挙動にフォールバック
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # データセットの属性 `is_text_only` (List[bool]) を取り出す
        is_text_only = self.train_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
        ) #こうして返される NoTextOnlyBatchSampler が、DataLoader 側でバッチを組む際に使われます。
    
    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        is_text_only = eval_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.eval_batch_size, #評価用バッチサイズ を args.eval_batch_size で取る
            world_size=self.args.world_size, #world_size は GPU数のみ（評価フェーズでは勾配蓄積を考慮しない）
            is_text_only=is_text_only,
        )


#LoRA 微調整対象の線形層名を自動収集するための関数
#モデルのモジュール一覧（名前→モジュールオブジェクトの辞書）から、指定した文字列を名前に含み、かつ torch.nn.Linear 型のレイヤーだけを抽出して名前リストとして返すものです。主に LoRA（Low-Rank Adaptation）の対象となる線形層名を自動で集めるの
def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    cls = torch.nn.Linear #torch.nn.Linear 型だけを残す
    lora_module_names = set()
    for name, module in named_modules.items():
        # 2-1. 名前に target_modules のいずれかが入っているか？
        if not any([module_name in name for module_name in target_modules]):
            continue

        # 2-2. かつ、torch.nn.Linear のインスタンスか？
        if isinstance(module, cls):
            lora_module_names.add(name)

    #特殊除外 (lm_head の除去)
    for name in list(lora_module_names):
        if 'lm_head' in name: # needed for 16-bit  16-bit モデルでは出力層を除きたい場合がある
            lora_module_names.remove(name)

    return list(lora_module_names) #最終的にリストで返却


#分散トレーニング中に、プロセス（GPU）ランク０ のみが標準出力にメッセージを出す
def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


#DeepSpeed ZeRO Stage 3 で分散配置されたパラメータを、一時的に GPU メモリに集約（gather）してから CPU に取り出す。ZeRO 未使用時や Stage 1/2 の場合は通常通りコピー。
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
#PEFT（LoRA など）で微調整対象のパラメータだけを抜き出し、さらに ZeRO Stage 3 下でも安全に CPU に集約した辞書を返す。
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


#DeepSpeed 使用時も含め、HuggingFace Trainer からモデル重みを安全にファイルに書き出す。
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
