import math
import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array #Vision トークンの数を計算するときに画素数を取得
from transformers.models.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessorKwargs #Processor に渡す追加パラメータの定義。
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator


logger = logging.get_logger(__name__)


# slightly different from https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/chat_template.json
# to include <|im_end|> of assistant's response as labels

###Chat テンプレート定義###
#「メッセージ（system／user／assistant）＋画像／動画トークン＋テキスト＋終了マーカー」をひとつのシーケンスに変換するためのテンプレート定義
template = (
    "{% for message in messages %}" #メッセージのループ開始
    "{{'<|im_start|>' + message['role'] + ' '}}" #各メッセージの先頭に <|im_start|>role （例：<|im_start|>user ）を挿入します。

    #画像→動画→テキストという順番を保つことで、モデル側で「まず映像情報を読み込んで、次に文章」という入出力順を統一
    #LLaVA-OneVisionのようなマルチモーダルモデルでは「ビジョン→テキスト」という流れで処理する
    "{# Render all images first #}"  #画像トークンの描画
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}" #message['content'] の中から type=='image' の要素だけを取り出し、対応する <image> トークンを先にすべて連続で出力。画像何枚分の画像トークンを挿入すべきか」がテンプレートで表現
    "{{ '<image>' }}"
    "{% endfor %}"

    "{# Render all video then #}" #動画トークンの描画
    "{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}" #同様に、type=='video' の要素だけを拾って <video> トークンを続けて出力。
    "{{ '<video>' }}"
    "{% endfor %}"

    "{# Render all text next #}" #テキストの描画（ユーザー vs. アシスタントで分岐）
    "{% if message['role'] != 'assistant' %}" #ユーザー（role != 'assistant'）の場合、テキスト要素を改行付きでそのまま出力し、最後に <|im_end|> を追加（下の共通終了マーカーで）。
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"

    "{% else %}" #アシスタントの場合、generation ブロックで囲むことで「この部分は生成ターゲット（モデルが予測すべき領域）」としてマスク。
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}" #{% generation %}...{% endgeneration %}で挟まれたトークンはモデルが生成すべき領域としてマスクされる部分
    "{{ '\n' + content['text'] }}"
    "{{'<|im_end|>'}}" #各テキスト要素のあとに <|im_end|> を出力して、「応答の終了」を明示
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    
    "{% if message['role'] != 'assistant' %}" #ユーザー応答の終了マーカー
    "{{'<|im_end|>'}}" #ユーザーのテキスト部分が終わったら、必ず <|im_end|> を付与し、「ここまでがユーザー発話」という区切り線を明確に。アシスタント側は上で <|im_end|> を入れているので書く必要なし。
    "{% endif %}"

    "{% endfor %}"
    "{% if add_generation_prompt %}" #ループ後：生成プロンプトの追加（オプション）
    "{{ '<|im_start|>assistant\n' }}" #学習データには「アシスタントの実際の回答全文」が含まれているが、それ以外の回答をモデルに出力させようとしたとき、つまり推論時にこのオプションを設定すると、最後に <|im_start|>assistant\n が追加される
    "{% endif %}"
)

"""
全体イメージ
学習時の例

<|im_start|>system 
You are a helpful assistant.
<|im_end|>
<|im_start|>user 
<image>
\nWhat is shown in this image?
<|im_end|>
<|im_start|>assistant 
{% generation %}
\nThis is a photo of a mountain landscape.  ←ここは正解の文章。実際はマスクさせて学習
<|im_end|>
{% endgeneration %}


といったトークン列が得られ、
<|im_start|>/<|im_end|> で各発話の開始・終了を明示し、
<image>/<video> でメディア挿入位置を示し、
アシスタント部分だけをマスクして学習させる（generation 範囲）
というフォーマットを一気に作り出せるようになっています。
"""


@register_collator("llava-onevision")
class LLaVAOnevisionDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]: #instances: Sequence[Dict] は DataCollator に渡される「バッチとしてまとめたい複数のサンプル」のリスト
        #Processor（画像／動画処理器）と Tokenizer の初期化引数を統合。
        output_kwargs = self.processor._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )

        #画像・動画の前処理 (バッチ内の全 images／videos を一括処理。)
        vision_inputs = dict()        
        # images
        #各サンプルの instance["images"]（リスト）をまとめて 1D リスト(flatten)
        images: List[List[PIL.Image.Image]] = [x for instance in instances for x in instance["images"]] #instanceが1サンプル、instancesが1バッチ
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"]))
        #image_processor の返り値
        # {"pixel_values": Tensor[B, C, H, W], "image_sizes": List[(H_i,W_i)], …}
        # これを vision_inputs に追加

        # videos
        videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]
        if len(videos) > 0:
            # ideally we should do padding here instead of forcing all videos to have the same length
            # but since currently hf implementation does not unpad videos or have corresponding 
            # attention masks, having padding will let the model train on padded frames
            assert len(set([x.shape[0] for x in videos])) == 1, "All videos must have the same number of frames"
            vision_inputs.update(**self.processor.video_processor(videos, return_tensors="pt", **output_kwargs["videos_kwargs"]))

        # some parsing
        #会話データの整形
        images = [instance["images"] for instance in instances]
        videos = [instance["videos"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        
        # constants
        max_len = self.tokenizer.model_max_length #トークナイザーが持つ、そのモデルで許容される最大入力長
        image_token_id = self.config.image_token_index # テンプレート中の <image> トークンに対応するトークンID
        video_token_id = self.config.video_token_index # テンプレート中の <video> トークンに対応するトークンID
        vision_feature_select_strategy = self.processor.vision_feature_select_strategy # 画像特徴の選択戦略
                                                                                       #vision_feature_select_strategy は Processor 側の設定で、
                                                                                       #"default"：ViT 等で出力される特徴ベクトルの数から special token（CLSなど）を除外する
                                                                                       #"first"：パラメータ変更がない代わりに全ベクトルを使う
                                                                                       # などの挙動を切り替えます。

        # construct input_ids and labels
        #目的：後でバッチ内の各サンプルごとに生成した input_ids（入力トークン列）と labels（学習用ラベル）を順に格納していくためのリスト
        input_ids = []
        labels = []
        
        #サンプルごとのループ開始
        #system_prompt：システム指示（例：役割説明）が入っていれば取り出し
        #cur_images / cur_videos：このサンプルに対応する画像／動画のリスト
        #cur_convs：文字列の会話履歴リスト（"<image>"／"<video>" を含むテキスト）
        for system_prompt, cur_images, cur_videos, cur_convs in zip(system_prompts, images, videos, conversations):
            cur_num_images = 0
            cur_num_videos = 0
            cur_input_ids = []
            cur_labels = []

            cur_text = [] #テンプレートに渡す「構造化メッセージ」のリストをここに蓄積
            #システムプロンプトの追加（任意）
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            #会話履歴（cur_convs）の分割処理
            #会話リストはユーザー／アシスタントが交互に入っている想定
            for i, text in enumerate(cur_convs):
                if i % 2 == 0: #i が 偶数 → ユーザー発話
                    # ① <image> の数をカウント
                    num_images = len([m.start() for m in re.finditer("<image>", text)]) #re.findall で文字列中の <image>／<video> を全検索。
                    cur_num_images += num_images

                    # ② <video> の数をカウント
                    num_videos = len([m.start() for m in re.finditer("<video>", text)])
                    cur_num_videos += num_videos

                    # .strip(): whitespaces and newlines are handled by chat_template
                    # ③ タグを取り除き、純粋なテキスト部分を抽出
                    text = text.replace("<image>", "").replace("<video>", "").strip()

                    # ④ 構造化メッセージを作成
                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_images + \
                            [{"type": "video"}] * num_videos
                    })
                else: #i が 奇数 → アシスタント発話
                    # ここでは画像／動画トークンを想定せず、純粋に「回答テキストのみ」を content に入れています。
                    cur_text.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}]
                    })
                
            #メディア数のチェック
            #実際に渡された cur_images（PIL 画像オブジェクト） と、テキスト中でカウントした <image> の数が合わないと一貫性を欠くため、ここでエラーを出します。
            assert len(cur_images) == cur_num_images, "Not all images were used"
            assert len(cur_videos) == cur_num_videos, "Not all videos were used"
            
            #テンプレート適用＋トークナイズ
            #先ほど組み立てた cur_text をテンプレート（template）に流し込み、最終的なトークン列（ひとつの長い文字列）を生成
            temp = self.processor.apply_chat_template(
                cur_text,
                chat_template=template,
                add_generation_prompt=False, #Trueのときテンプレート末尾に <|im_start|>assistant という文字列（「アシスタントの発話開始マーカー」）を追加する、
                                             #学習データ（トレーニング）では既にモデルに見せる「正解のアシスタント応答全文」が含まれているので、ここは False にして追加しません。
                                             #一方、推論モードで「ここから続きを自動生成してほしい」とモデルに cue（合図）を与えるには、会話履歴の末尾に <|im_start|>assistant\n を付けておく必要があります。
                tokenize=True, #文字列ではなく、すぐにトークナイザーでトークンID列に変換する
                return_assistant_tokens_mask=True, #アシスタント生成部分を示すマスク配列（0/1）を返す
                return_dict=True, #	辞書形式（dict）で返す
                return_tensors="pt", #NumpyではなくTensorで返す
                truncation=False # the assistant tokens mask seems wrong when truncation is enabled ここでは意図的に切り捨てしない（assistant_masks がずれるため）
            )

            #呼び出し後の temp（辞書）には、少なくとも以下が含まれています。
            # "input_ids" 形状 (1, seq_len) の torch.LongTensor。 <|im_start|> など特殊トークンを含む、最終的なトークンID列
            # "attention_mask" (1, seq_len) の 0/1 マスク（PAD でない位置が 1）
            # "assistant_masks" (seq_len,) の 0/1 リストまたは配列。1 の箇所がモデルに「ここを予測させる（生成させる）」部分、0 が「入力として与えるだけ＆損失計算から除外」部分
            # その他
            # token_type_ids, special_tokens_mask など、モデルによっては追加情報を返す場合あり
            cur_input_ids = temp["input_ids"] #最終的にモデルの Embedding テーブルに突っ込むトークンID列
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0) #cur_assistant_masks：学習時にこのマスクを使って、assistant_masks==0 の位置の損失を IGNORE_TOKEN_ID に置き換え、ユーザー質問部分には勾配が回らないようにします。

            # expand vision tokens
            #画像トークンの拡張
            if len(cur_images) > 0:
                # (1) 画像前処理を再度実行して特徴サイズを得る
                image_inputs = self.processor.image_processor(cur_images, return_tensors="pt", **output_kwargs["images_kwargs"])

                image_sizes = image_inputs["image_sizes"] # 各画像の〔元高さ, 元幅〕リスト
                # (2) Transformer のビジョン部分で使う「標準パッチサイズ」を取得
                #    → 実際にパッチ化して得られる特徴マップの H × W
                height, width = get_image_size(
                    to_numpy_array(image_inputs["pixel_values"][0][0]), 
                    channel_dim=output_kwargs["images_kwargs"].get("data_format")
                )

                # (3) 各画像が何トークン分の特徴になるかを計算
                num_image_tokens_list = []
                for image_size in image_sizes:
                    orig_height, orig_width = image_size
                    # _get_number_of_features は例えば ViT の (orig_h/patch × orig_w/patch + special token) を返す
                    num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                    # “default” 戦略では先頭の特殊トークン（例：CLS 相当）を除くため 1 を引く
                    if vision_feature_select_strategy == "default":
                        num_image_tokens -= 1
                    num_image_tokens_list.append(num_image_tokens)

                # (4) cur_input_ids 中の <image> トークン位置を見つけ、対応する n をセット
                repeat = torch.ones(cur_input_ids.shape[1], dtype=torch.long) # デフォルトは「1 回ずつ繰り返す」
                repeat[torch.where(cur_input_ids == image_token_id)[1]] = torch.tensor(num_image_tokens_list, dtype=torch.long)
                # (5) repeat_interleave で各 <image> を n 個に展開
                #例えば repeat = [1, 1, 256, 1, ...] なら、3 番目のトークン（<image>）だけ 256 回に拡張され、
                #モデルの入力トークン列（cur_input_ids, cur_assistant_masks）が「1トークン→n特徴ベクトル」に対応づけられます。
                cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)
            
            #動画トークンの拡張
            if len(cur_videos) > 0:
                # (1) 動画前処理を実行してフレーム数と解像度を取得
                video_inputs = self.processor.video_processor(cur_videos, return_tensors="pt", **output_kwargs["videos_kwargs"])

                one_video = to_numpy_array(video_inputs["pixel_values_videos"][0])
                height, width = get_image_size(
                    one_video[0], 
                    channel_dim=output_kwargs["images_kwargs"].get("data_format")
                )
                num_frames = one_video.shape[0]  # frame dim is always after batch dim  フレーム数
                # (2) 空間パッチ数を計算
                patches_height_width = int(math.sqrt(self.processor.num_image_tokens))
                pooled_height_width = math.ceil(patches_height_width / 2)
                # フレーム × (空間プーリング後のパッチ数)² +1(newline token)
                num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline token

                # (3) <video> の位置を見つけ、展開回数を num_video_tokens に
                repeat = torch.where(cur_input_ids == video_token_id, num_video_tokens, 1).squeeze()
                # (4) Image と同様に repeat_interleave で拡張
                cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)

            # manual truncation
            #この部分は「シーケンス長制限」「ラベル準備」「質問トークンのマスク化」をまとめて行なうセクション
            #手動トランケーション（長すぎる場合の切り捨て）
            #トークナイザーが返す input_ids の長さがモデルの許容上限（max_len）を超えてしまうと、モデルが動かないか、attention マスクがずれる原因になります。
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len] #cur_input_ids の２次元目（シーケンス長）が max_len を超えていたら、先頭から max_len トークンだけを残して切り捨て。
                cur_assistant_masks = cur_assistant_masks[:, :max_len] #同様に、対応するアシスタント生成マスク (cur_assistant_masks) も同じ位置で切り揃えます。
            #ラベルのクローン
            #モデルの学習時には、input_ids と同じトークン列を「正解ラベル（labels）」として与え、生成すべき部分（アシスタント応答）に対して次のトークンを予測するようにします。
            cur_labels = cur_input_ids.clone()

            #質問トークン（ユーザー発話）のマスク化
            #mask_question_tokens が True の場合
            #cur_assistant_masks は「生成ターゲット（アシスタント応答）の位置」を True／1 で示すブールマスク。
            #それ以外（ユーザー発話や system メッセージ）は False／0。
            if self.mask_question_tokens:
                assert cur_labels.shape == cur_assistant_masks.shape, "Label and mask shapes do not match"
                cur_labels = torch.where(cur_assistant_masks, cur_labels, self.IGNORE_TOKEN_ID) #torch.whereマスクが True の位置には元の cur_labels の値（トークンID）を残し、マスクが False の位置には self.IGNORE_TOKEN_ID をセット。
            
            #input_ids と labels の形がずれているとバッチ化や損失計算でエラーになるため、念のため形状が完全に一致しているかを強制チェック。
            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"

            # padding
            #このパディング部分では、「シーケンス長が max_len に満たない場合に、末尾を埋めて長さを揃える」処理を行っている
            if cur_input_ids.shape[1] < max_len:
                #(1) input_ids のパディング
                cur_input_ids = torch.cat([
                    cur_input_ids, # (a) もともとのトークン列
                    # (b) PAD_TOKEN_ID で埋めた行列
                    torch.full( # shape: (batch_size=1, pad_length)
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]), #(1, max_len - seq_len) の形で、 全要素が PAD_TOKEN_ID のテンソルを作成。
                        self.PAD_TOKEN_ID, # 埋める値（例：0）
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1) # トークン列の最後（dim=1）に連結
                #(2) labels のパディング
                cur_labels = torch.cat([
                    cur_labels, # (a) もともとのラベル列
                    # (b) IGNORE_TOKEN_ID で埋めた行列
                    torch.full(
                        (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        self.IGNORE_TOKEN_ID,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    )
                ], dim=1)

            # 各サンプルごとに作った (1, max_len) テンソルをリストに追加
            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

        # ループ後、一気に結合して (batch_size, max_len) に
        #各サンプル (1, L) → (B, L) にまとめる
        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        # vision_inputs（pixel_values や video テンソルなど）とまとめて返す
        #vision_inputs（画像／動画用テンソル）
        # input_ids, labels, attention_maskをすべてまとめた辞書を返し、Transformer モデルに渡せる完全なバッチが完成します。
        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID), # attention_mask を再計算
        )
    