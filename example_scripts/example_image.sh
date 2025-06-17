#!/bin/bash

###分散トレーニング設定###
NUM_GPUS=1 # ノードあたりのGPU数
#torchrunに渡す分散学習用の共通引数。内部的にPyTorchの分散処理を初期化するために用いられる。
#--nnodes=1 「分散トレーニングを行うノード数」を 1 に指定。単一マシンでのみ実行することを意味します。
#--nproc_per_node ${NUM_GPUS} 「各ノードあたりのプロセス数＝GPU 数」を指定しています。
#--rdzv_backend c10d 「ラウンドロビン方式での rendezvous（通信グループ結成）に使うバックエンド」を c10d に指定。PyTorch 標準の通信バックエンドです。 
#--rdzv_endpoint localhost:0 「rendezvous（通信初期化）のエンドポイント」を localhost:0（ランダムなポート）に指定。単一マシン上で分散する場合にこのように書くことが多いです。
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

###モデル・データ・メディアパスの指定###
# arguments that are very likely to be changed
# according to your own case
MODEL_ID=llava-1.5-7b                                   # model id; pick on by running `python supported_models.py` ファインチューニング対象のモデル名を指定しています。
TRAIN_DATA_PATH=./example_data/celeba_image_train.json  # path to the training data json file 学習用データセット（JSON 形式）のファイルパスを指定しています。
EVAL_DATA_PATH=./example_data/celeba_image_eval.json    # path to the evaluation data json file (optional) 検証（evaluation）用データセットの JSON ファイルパスを指定しています。
IMAGE_FOLDER=./example_data/images                      # path to the image root folder; if provided, the image paths in the json should be relative　画像ファイルを格納しているルートフォルダを指定。「JSON の中で \"image\": \"xxx.jpg\" のように相対パスを書いている場合は、このフォルダを起点にファイルを探します」。
VIDEO_FOLDER=./example_data/videos                      # path to the video root folder; if provided, the video paths in the json should be relative　動画ファイルを格納しているルートフォルダ。今回の例では画像のみ（CelebA などの顔画像を想定）をファインチューニングする想定のため、動画は空でもよいですが、変数だけ定義されています。実際に動画を使う場合はここに動画フォルダを置きます。
NUM_FRAMES=8                                            # how many frames are sampled from each video 動画を扱う場合、各動画から何フレーム抽出して入力テンソルを作るかを指定します。動画モデルを使う際に必要なオプションです。（画像モデルの場合は無視されるか、デフォルト扱いになります）

###Vision（画像／動画）関連の学習フラグ###
TRAIN_VISION_ENCODER=False                              # whether train the vision encoder 「ビジョンエンコーダを更新（学習）するかどうか」を指定します。
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)  「ビジョンエンコーダにも LoRA（低ランク分解による微調整）を適用するかどうか」を指定。
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported) 「ビジョンエンコーダの出力を LLM に渡す前にマッピングするプロジェクタ層を学習するかどうか」を指定します。

###LoRA（Low-Rank Adaptation）設定###
USE_LORA=True                                           # whether use lora for llm 「LLM（大規模言語モデル）に対して LoRA を適用して微調整するか」を指定。
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True 「量子化に対応した LoRA（Q-LoRA）方式で学習を行うか」を指定します。
LORA_R=8                                                # the lora rank (both llm and vision encoder) LoRA の「ランク（低ランク分解後の行列サイズ）」を指定。
LORA_ALPHA=8                                            # the lora alpha (both llm and vision encoder) LoRA のスケーリングファクター。通常はランクと同じ値にすることが多いです。LoRA の計算式（W = W₀ + α·(A·B)）における α の値を決めます。

###実行結果を管理するための Run ID###
RUN_ID=${MODEL_ID}_lora-${USE_LORA}_qlora-${Q_LORA}     # a custom run id that determines the checkpoint folder and wandb run name 
                                                        # これをもとに出力ディレクトリ（--output_dir ./checkpoints/$RUN_ID）や、W&B（Weights & Biases）での run 名 (--run_name $RUN_ID) が決まります。

###DeepSpeed ステージ＆バッチサイズなどハイパーパラメータ###
DS_STAGE=zero3                                          # deepspeed stage; < zero2 | zero3 > DeepSpeed のメモリ最適化ステージを指定。
                                                        # zero2 または zero3 を選択でき、zero3 は最も Aggressive（積極的）にパラメータを分散配置して GPU メモリを節約します。
                                                        # zero3 のほうがメモリ消費を抑えられますが、通信オーバーヘッドが増えるので注意が必要です。
PER_DEVICE_BATCH_SIZE=2                                 # batch size per GPU 各 GPU（各プロセス）あたりのバッチサイズ を 2 に設定。たとえば NUM_GPUS=4 の場合、合計バッチサイズは 2 × 4 = 8 になります。
GRAD_ACCUM=1                                            # gradient accumulation steps 勾配累積ステップ数。
NUM_EPOCHS=5                                            # number of training epochs データセットを何周（エポック）学習するかを 5 に設定しています。

LR=2e-5                                                 # learning rate 学習率（learning rate）
MODEL_MAX_LEN=1024                                       # maximum input length of the model モデルが受け取れる入力トークン列の最大長を 1024 トークンに指定。もし会話履歴が長いときは途中で切り捨てられるので、必要に応じて大きめに設定しますが、メモリ消費も増える点に注意。


torchrun $DISTRIBUTED_ARGS train.py \
    --model_id $MODEL_ID \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA
    

<< COMMENTOUT
A. 基本実行部分
torchrun $DISTRIBUTED_ARGS train.py \
分散トレーニングを起動するエントリポイント。
$DISTRIBUTED_ARGS には前述の --nnodes=1 --nproc_per_node=1 ... などが展開されます。
train.py はメインのファインチューニングスクリプトです（リポジトリ配下にある学習ループ本体）。

B. モデル・データ関連
--model_id $MODEL_ID
ファインチューニング対象のモデル（Hugging Face Hub 上のモデル名やローカルパス）を指定。
例：llava-1.5-7b。

--data_path $TRAIN_DATA_PATH
学習用データセット（JSON）のファイルパスを指定。

--eval_data_path $EVAL_DATA_PATH
検証用データセット（JSON）のファイルパスを指定。

--image_folder $IMAGE_FOLDER
画像データを格納したルートディレクトリを指定。

--video_folder $VIDEO_FOLDER
動画データを格納したルートディレクトリを指定。

--num_frames $NUM_FRAMES
動画からサンプリングするフレーム数を指定。画像のみの場合は無視されますが、指定すると動画モデルはこの数だけフレームを使ってトレーニングします。

--output_dir ./checkpoints/$RUN_ID
チェックポイントや最終モデルを保存するディレクトリを指定。たとえば ./checkpoints/llava-1.5-7b_lora-True_qlora-False が出力先になります。

C. ロギング・実験管理関連
--report_to wandb
学習状況（損失や学習率など）のログを Weights & Biases（wandb）に送る設定。後述の --run_name で名前を付けると、W&B 上でも比較しやすくなります。

--run_name $RUN_ID
W&B や TensorBoard 上で表示される実験名を指定。たとえば llava-1.5-7b_lora-True_qlora-False のように一意に決めておくと、複数実験を一覧しやすい。

D. DeepSpeed・精度設定関連
--deepspeed ./ds_configs/${DS_STAGE}.json
DeepSpeed の設定ファイルを JSON で指定。例では ds_configs/zero3.json を使う。
JSON 内には zero_optimization のステージ（zero2/zero3）、fp16/bf16 の有効化、バッチ分割の方法など、DeepSpeed 固有の詳細設定が書かれています。

--bf16 True
bf16（Brain Floating Point 16）精度を有効化。学習速度とメモリ効率を向上させつつ、float32 相当の精度を保ちやすい。環境によっては --fp16 True（半精度 float16）を使う場合もあります。

--tf32 True
NVIDIA Ampere 以降の GPU で使える TF32（TensorFloat-32）を有効化。マトリクス計算を高速化しつつ、精度をそこそこ保つモードです。

E. トレーニングスケジュール・チェックポイント関連
--num_train_epochs $NUM_EPOCHS
エポック数を 5 に設定。データセットを 5 周する。

--per_device_train_batch_size $PER_DEVICE_BATCH_SIZE
学習時のミニバッチサイズを 2（GPU あたり）に指定。NUM_GPUS=1 の場合、合計バッチサイズは 2。

--per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE
検証時のバッチサイズも同じく 2。

--gradient_accumulation_steps $GRAD_ACCUM
勾配累積ステップ数を 1 に指定。1 ステップごとに更新する設定なので、実質的にはバッチサイズ 2 のまま。

--eval_strategy "epoch"
「1 エポックごとに検証を行う」ことを指定。"step" にすれば指定ステップ間隔で検証も可能ですが、この例ではエポック単位。

--save_strategy "epoch"
「1 エポックごとにモデルのチェックポイントを保存する」設定。

--save_total_limit 1
保存するチェックポイントの最大数を 1 に制限。過去のチェックポイントを自動で消してくれるため、ディスク容量節約に役立つ。

--logging_steps 1
「何ステップごとにログを標準出力に出力するか」を指定。1 なので、毎ステップで損失や学習率などを表示（ログが多く出るが、細かく監視したいときに便利）。

F. オプティマイザ・スケジューラ関連
--learning_rate ${LR}
学習率を 2×10^−5 に設定。

--weight_decay 0.
重み減衰（L2 正則化）の係数を 0 に設定（正則化なし）。必要に応じて小さめ（例：1e-2）にすると過学習を抑えられます。

--warmup_ratio 0.03
総ステップ数のうち 3％ をウォームアップに使い、徐々に学習率を LR まで上げる設定。
たとえば総ステップが 1000 なら 30 ステップかけて線形増加させ、以降はスケジューラに従って減衰させます。

--lr_scheduler_type "cosine"
AdamW 等のオプティマイザにおける学習率スケジューラを「コサイン減衰（cosine）」に指定。ウォームアップ後に徐々に減衰していく典型的なスケジュールを使います。

G. モデル入力長・省メモリ機能関連
--model_max_length $MODEL_MAX_LEN
モデルの許容する入力トークン最大長を 1024 に指定。これを超えた分はトランケートされます。

--gradient_checkpointing True
勾配チェックポイント（Gradient Checkpointing）を有効化。通常はアクティベーションをすべて保持しますが、チェックポイントをかけると中間層の出力を再計算することでメモリを節約できます。
大きいモデルを少ないメモリで学習したいときに非常に有効ですが、逆伝播のたびに一部再計算が入るため計算コストは若干増えます。

--dataloader_num_workers 4
PyTorch の DataLoader で画像・テキストのデータを読み込む際に使う Worker プロセス数を 4 に指定。CPU サイドでのデータ読み込み・前処理を並列化して、GPU を待たせずにデータを読み込むようにします。

H. Vision／LoRA フラグの反映
--train_vision_encoder $TRAIN_VISION_ENCODER
先ほどの変数 TRAIN_VISION_ENCODER=False をここで渡し、「ビジョンエンコーダ部分を更新するかどうか」を学習スクリプトに伝えます。

--use_vision_lora $USE_VISION_LORA
USE_VISION_LORA=False なので、ビジョン部に LoRA は適用されません。もし True にすると、vision encoder 部分も LoRA で微調整します。

--train_vision_projector $TRAIN_VISION_PROJECTOR
ビジョンエンコーダ→LLM へ接続するためのプロジェクタ層を学習するかどうかを指定。

--use_lora $USE_LORA
USE_LORA=True なので、LLM 部分は LoRA で学習します（通常より少ないパラメータ更新で済む）。

--q_lora $Q_LORA
Q_LORA=False のため、通常の LoRA（非量子化モード）で学習します。もし True にすると QLoRA モードで量子化＋LoRA を併用します。

--lora_r $LORA_R
LoRA のランク（8）を渡しています。

--lora_alpha $LORA_ALPHA
LoRA のスケーリングファクター（8）を渡しています。
COMMENTOUT
