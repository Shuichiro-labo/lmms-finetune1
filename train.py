import os
os.environ["WANDB_PROJECT"]= "lmms-ft" #Weights & Biases (wandb) でログを取る際に、プロジェクト名を "lmms-ft" にあらかじめ指定
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml

from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
#from transformers import Trainer, deepspeed
from transformers import Trainer
from transformers.integrations import deepspeed


#自作モジュール
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments #コマンドライン引数を dataclass 化して定義しているファイル。
from collators import COLLATORS #各モデルファミリ専用の DataCollator をあらかじめ登録している辞書。たとえば LLaVA 系なら「画像＋テキスト＋マスクトークン＋パディング」を一度に処理するメソッドを持つクラス、Qwen-VL 系ならその専用クラスが紐づいています。
from datasets import LazySupervisedDataset #JSON データを遅延的（Lazy）にロードし、PyTorch Dataset として扱うクラス。
from loaders import LOADERS #モデルファミリごとに「モデル＋トークナイザー＋画像／動画プロセッサ」を読み込むクラス群を登録している辞書。
from supported_models import MODULE_KEYWORDS #各モデルファミリごとに「vision encoder 部分のキー名」「vision projector 部分のキー名」「LLM 部分のキー名」「その他凍結すべきモジュール」などを定義。
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
) #utils に含まれる関数群：進捗やログを rank0（メインプロセス）でしか出力しないようにする rank0_print、LoRA 対象のモジュール名を取り出す find_all_linear_names、DeepSpeed Stage3＋PEFT の局所オフセットを扱うユーティリティ、カスタムサンプラーを組み込んだ TrainerWithCustomSampler など。


###MLLMの効率的にファインチューニングするためのファイル###

def train():
    ###引数のパースとダンプ###

    #コマンドラインから渡された --model_hf_path や --data_path、--output_dir、--lora_r といったオプションを自動でパースし、各インスタンスに詰めます。
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    #どの引数で学習を開始したのか後からすぐにわかるように、output_dir/arguments/ フォルダ配下に下記ファイルを出力します：
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    #asdict() で dataclass の中身を辞書化し、yaml.dump でファイルに書き出しています。
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w")) #（モデル設定）
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w")) #（データ設定）
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w")) #（学習用ハイパーパラメータ）
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w")) #（LoRA 関連の設定）




    ###計算精度と分散タイプの判定###
    #--fp16 True を渡していれば torch.float16、--bf16 True を渡していれば torch.bfloat16、両方指定がなければ torch.float32 を使用する、という優先度で決定しています。
    #この compute_dtype は後の量子化設定（QLoRA）やモデルロード時に利用されます。
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    #DeepSpeed ＋ QLoRA の組み合わせチェック
    #もし --deepspeed（DeepSpeed の設定ファイルを参照するオプション）を指定しており、かつ --q_lora True（4bit 量子化 LoRA）を指定している場合、内部的に training_args.distributed_state.distributed_type を DeepSpeed に強制セットします。
    #これは “QLoRA を使うときは DeepSpeed の ZeRO Stage ３ 以外はサポートしていない” といった制限を担保するためです。
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED




    ###デバイスマップ（device_map）の設定（QLoRA 用）###
    #QLoRA（4bit LoRA）のとき、モデルをどの GPU（または CPU）に配置するかをあらかじめ指定するための辞書
    #{"": <GPU_INDEX>} とすることで、「モデル全体を <GPU_INDEX> 番のデバイスに載せる」という意味になります。
    #もし WORLD_SIZE（分散トレーニングで総プロセス数）が 1（＝単一 GPU 実行）であれば device_map=None のままにし、デフォルトの動作（自動的に GPU0 を使うなど）に任せます。
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        #FSDP（FullyShardedDataParallel）や ZeRO3 のチェック
        #--fsdp オプションを渡していたり、DeepSpeed ZeRO Stage3 が有効なときは、QLoRA とは「同時に使えない」ためエラーを出して実行を止めます。
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")




    # llm quantization config (for q-lora)
    ###QLoRA 用の量子化設定（bnb_config）###
    # bnb_config
    # Hugging Face Transformers の BitsAndBytesConfig を使い、4bit 量子化を有効にする設定オブジェクトを作成します。
    # load_in_4bit=True とすることで、モデルの重みを 4bit 量子化して読み込むようになります。
    # bnb_4bit_compute_dtype には先ほど決めた compute_dtype（fp16／bf16／float32）を設定。計算時に必要な精度を担保します。
    # bnb_4bit_quant_type="nf4" は量子化方式の一種。通常は "nf4"（Normal Float 4）か "fp4" が選べます。
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4", 
        )
    



    # load model, tokenizer, processor
    ###モデル・トークナイザー・プロセッサのロード###
    # LOADERS[model_args.model_family_id]
    # model_family_id（例：llava,qwen-vl など）に対応する「モデル・トークナイザー・プロセッサを読み込むクラス」をあらかじめ辞書で登録しています。
    # 具体的には、LLaVA 系なら LLaVA 用のカスタム Loader、Qwen-VL 系ならその専用 Loader を使って、「モデル重みのロード→トークナイザー初期化→マルチモーダル用プロセッサの生成→設定オブジェクトの生成」を行います。
    rank0_print("Loading model, tokenizer, processor...")
    loader = LOADERS[model_args.model_family_id](
        #引数として渡すもの
        model_hf_path=model_args.model_hf_path, #Hugging Face Hub 上のモデル名か、あるいはローカルに保存した重みディレクトリのパス。
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype, #先ほど決めた float16／bf16／float32 のいずれか。
        bnb_config=bnb_config, #もし QLoRA を使うなら 4bit 量子化設定オブジェクト。
        use_flash_attn=training_args.use_flash_attn, #FlashAttention を使うかどうか。高速化のためのオプション
        device_map=device_map, #マルチ GPU や QLoRA 時に「どの GPU にモデルを載せるか」を制御する辞書。
    )
    model, tokenizer, processor, config = loader.load() #loader.load() の返り値
                                                        # model（torch.nn.Module）
                                                        # tokenizer（transformers.PreTrainedTokenizer）
                                                        # processor（マルチモーダルモデル向けに、画像/動画フレームの前処理を行うユーティリティ）
                                                        # config（モデル設定情報を格納した辞書的オブジェクト）
    tokenizer.model_max_length = training_args.model_max_length #入力トークンの最大長をトークナイザー側にも反映
                                                                #text をトークン化した結果が 1024 トークンを超えていれば、自動的に後ろから切り捨てして 1024 トークンに揃え
                                                                #1024 トークン未満であれば、パディングして 1024 トークン分の長さに揃えるという動作を保証してくれます。

    if training_args.gradient_checkpointing: #Trueのときはモデルに勾配チェックポイントを有効化します。
        model.enable_input_require_grads()




    # freeze certain params
    ###ビジョン関連モジュールの凍結（あるいは全訓練対象扱い）###
    # "vision_encoder"：Vision エンコーダ（例：CLIP-ViT 部分や ViT-B）を構成するサブモジュール名のリスト
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"] #あるモデルのVisionEncoderを取り出している
    ## training_args.train_vision_encoder=False（デフォルト）なら、vision encoder 関連のすべてのサブモジュールの requires_grad を False にして完全に凍結
    if not training_args.train_vision_encoder: #if not はtraining_args.train_vision_encoderがFalseのとき、実行される。not False＝True
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys: #ループ内で、リストにあるサブモジュール名（例えば "vision_encoder.patch_embed" や "vision_encoder.encoder.layer" など）を一つずつ取り出す
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False) #各サブモジュールに対して勾配を計算しないよう設定

    # "vision_projector"：Vision エンコーダの出力を LLM 用にマッピングする「Projection 層」のサブモジュール名のリスト
    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    # 同様に training_args.train_vision_projector=False なら projection 部分も凍結。
    if not training_args.train_vision_projector:
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # we will just freeze these
    # "others"（オプション）：上記以外のマルチモーダル関連モジュール（例：フレームを整形する vision_resampler、画像をテキスト用に変換する image_newline など）があればここに列挙。
    # "others" キーが存在すれば、そこに列挙されたモジュールもすべて同じく凍結。
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]: 
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)




    # lora preparation
    ###LoRA（および QLoRA）設定###
    # "llm"：LLM 部分（例：Llama2 本体や Qwen-VL 本体）のサブモジュール名のリスト
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    # どのケースで LoRA を使うか否かの判定
    # lora_args.use_lora が True なら「LLM に対して LoRA を適用する」
    # training_args.train_vision_encoder and lora_args.use_vision_lora が True なら「Vision Encoder 部分にも LoRA を適用する」
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)): #if not はtraining_args.train_vision_encoderがFalseのとき実行される、LoRAは使われない
        rank0_print("No LoRA enabled...")
    #モデル中の全サブモジュールを「名前 → モジュールオブジェクト」の辞書にしておきます。
    #これを使って、どこに LoRA 用の低ランク分解モジュールを挿入すべきか調べやすくします。        
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = [] #（LoRA を適用したい線形層の名前リスト）と
        full_modules = [] #（フルファインチューニングしたいモジュール名リスト）を空で用意。
        #ここで lora_modules に入った線形レイヤー名が、PEFT（LoRA）設定の target_modules となり、
        #full_modules に入ったモジュール名がフルファインチューニングされる用として modules_to_save に指定されます。

        #Vision Encoderの部分
        #「train_vision_encoder=True かつ use_vision_lora=True　Vision Encoder 部分にも LoRA を適用する」
        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            rank0_print("LoRA for vision encoder enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
        #「train_vision_encoder=True　Vision Encoder 部分をフルファインチューニング、だがLoRAは使わない」「train_vision_encoder=False　（何もしない（＝Vision エンコーダ凍結）」
        elif training_args.train_vision_encoder:
            rank0_print("Vision encoder will be fully trained...")
            full_modules.extend(vision_encoder_keys)

        #LLMの部分
        #use_lora=True　LLM（言語モデル）部分のすべての線形レイヤーを LoRA 対象にする
        if lora_args.use_lora:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        #use_lora=False LLM 部分をフルファインチューニングする
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)
        
        #Projection 層の部分
        #train_vision_projector=True の場合のみ、Vision→LLM 間の「プロジェクタ層」をフルファインチューニング対象に追加。
        if training_args.train_vision_projector:
            rank0_print("Vision projector will be fully trained...")
            full_modules.extend(vision_projector_keys)
    
        # LoraConfig(...) の主要オプション
        lora_config = LoraConfig(
            r=lora_args.lora_r, # r：LoRA のランク
            lora_alpha=lora_args.lora_alpha, # lora_alpha：スケーリング係数
            target_modules=lora_modules, # target_modules：LoRA を挿入する「線形層の名前リスト」
            modules_to_save=full_modules, # modules_to_save：QLoRA/DeepSpeed Stage3 と併用するときに「学習後も保存すべき（更新情報を保持しておきたい）モジュール名」を列挙
            lora_dropout=lora_args.lora_dropout, # lora_dropout：LoRA 用のドロップアウト率
            bias=lora_args.lora_bias, # bias：LoRA 用バイアスの扱い方（"none", "all", "lora_only" などから選択）
            task_type="CAUSAL_LM", # task_type="CAUSAL_LM"：因果言語モデル（Chat 型）のタスクであることを指定
        )

        # QLoRA（量子化 LoRA）の前処理
        # if lora_args.q_lora: のときに prepare_model_for_kbit_training(model, ...) を呼ぶことで、内部的にモデルを 4bit 量子化向けの構造に変換します。
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        
        # 以降、get_peft_model をかけることで、「4bit 重み＋LoRA 更新可能なサブモジュール」の形にモデルが書き換わります。
        model = get_peft_model(model, lora_config)
        # LoRA 設定をもとに、元のモデルに PEFT 用のラッパーを適用して「LoRA 用の追加パラメータ」や「勾配フック」を差し込みます。
        # これにより、元のモデル重み自体は凍結しつつ、「低ランク分解用の枝分かれ重みだけを更新」できるようになります。




    # print trainable parameters for inspection
    ###更新可能パラメータの確認###
    # ここでは実際に requires_grad=True になっているパラメータ（＝学習時に更新されるパラメータ）を列挙し、ログに出力してユーザに「本当に意図したモジュールだけ更新される状態になっているか」を確認させています。
    # rank0_print を使うことで、分散環境下でもメインプロセス（rank 0）のみが出力し、他プロセスが重複して同じログを出さないようにします。
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")




    # load data
    ###データセットの読み込み###
    #Eager：初期化時にすべて読み込んで保持する
    #Lazy:必要になったときだけファイルを読み込む
    #大量の画像・テキストを一度にメモリに読み込まずにミニバッチ単位で前処理ができる
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, # data_args.data_path：学習用 JSON ファイルのパス
        image_folder=data_args.image_folder, # data_args.image_folder / video_folder：JSON 中のパスを解釈するルートディレクトリ
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames, # data_args.num_frames：動画サンプリングを行う際に何フレーム抜き出すか
        model_family_id=model_args.model_family_id, # model_family_id：モデルファミリ（例：llava）を教えることで、内部でどのプロセッサを呼び出すかが切り替わる
        user_key=data_args.user_key, # user_key / assistant_key：JSON 中の会話エントリで「ユーザー」と「アシスタント」を表すキー文字列（デフォルト "human" / "gpt" など）を指定
        assistant_key=data_args.assistant_key
    )
    # --eval_data_path を渡していれば評価用データセットを生成
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            video_folder=data_args.video_folder,
            num_frames=data_args.num_frames,
            model_family_id=model_args.model_family_id,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key
        )
    #そうでなければ None にして training_args.eval_strategy="no"（評価を一切行わないモード）に切り替え
    else:
        eval_dataset = None
        training_args.eval_strategy = "no"




    # data collator
    ###DataCollator（バッチ整形器）の準備###
    data_collator = COLLATORS[model_args.model_family_id](
        config=config, # config：モデル設定（トークナイザー、最大長、特殊トークン情報など）
        tokenizer=tokenizer, # tokenizer：テキストをトークン ID に変換するためのオブジェクト
        processor=processor, # processor：画像や動画フレームをテンソルに変換するためのユーティリティ
        mask_question_tokens=training_args.mask_question_tokens # mask_question_tokens：もし質問文中のトークンをマスクしたいなら True にしておくフラグ。
    )




    # trainer
    ###Trainer の生成と学習開始###
    trainer = TrainerWithCustomSampler(
        model=model, # model：LoRAや凍結などの前処理をすでに施した状態の torch.nn.Module
        args=training_args, # args：TrainingArguments（バッチサイズ、エポック数、DeepSpeed 設定、ログ頻度、勾配チェックポイント、バラメータ更新スケジュールなど）
        data_collator=data_collator, # data_collator：上で作成した DataCollator
        train_dataset=train_dataset, # train_dataset / eval_dataset：先ほど生成した Dataset インスタンス
        eval_dataset=eval_dataset,
    )
    trainer.train() # 学習ループを実行。
    trainer.save_state() # 学習中の内部ステート（オプティマイザ状態、スケジューラ状態、DeepSpeed のメタ情報など）を保存します。

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    #safe_save_model_for_hf_trainer はこのプロジェクト独自のユーティリティで、
    #DeepSpeed ZeRO3 で分散された重みを一箇所に集める
    #LoRA のアダプタだけでなく、modules_to_save 指定のフルパラメータも統合して保存する
    #QLoRA の 4bit 重みと LoRA をマージして推論用モデルを出力する
    #といった後処理を行います。




if __name__ == "__main__":
    print("[DEBUG] start of train.py reached")
    train()
