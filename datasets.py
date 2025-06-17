import av #PyAV (av)：動画ファイルからフレームをデコードするために使用。
import os
import json
from PIL import Image #Pillow (Image)：画像ファイルを開いて RGB 形式に変換するために使用。
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

#JSON で定義されたマルチモーダル（画像・動画＋会話）データを、遅延的（Lazy）に読み込みつつ PyTorch の Dataset として扱うための実装例


#モデルファミリによっては、画像を実際に PIL.Image オブジェクトとして読み込む必要があるものと、「パスのまま渡すだけ」でよいものがあります。
#この辞書で、各モデルがどちらを要求しているかを管理しています。
TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "llava-onevision": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
    "llama-3.2-vision": True,
}


#動画関連の関数
#PyAV の InputContainer から指定したフレーム番号だけをデコードし、(num_frames, H, W, 3) 形の NumPy 配列にまとめて返す
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0) #container.seek(0) で先頭にリセット
    start_index = indices[0] #container.decode(video=0) で全フレームを順次取得しつつ、
    end_index = indices[-1] #フレーム番号 i が indices リストに含まれる場合のみメモリに保持
    for i, frame in enumerate(container.decode(video=0)): #フレーム番号 i が indices リストに含まれる場合のみメモリに保持
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames]) #最終的に RGB24 フォーマットで NumPy に変換し、スタックして返却


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    #JSON ファイルを読み込み、各サンプルのメタ情報（画像／動画パス、会話リスト、system_prompt など）を list_data_dict に格納。
    def __init__(
        self, 
        data_path: str, 
        model_family_id: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r")) #data_path で指定された JSON をパースし、各サンプルの辞書（画像や動画パス、会話リスト、system_prompt などが入った dict）のリストを self.list_data_dict に格納します。
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.load_image = TO_LOAD_IMAGE[model_family_id] #モデルによっては画像を PIL で開かずパスだけ渡せばよいケースがあるため、そのフラグをここで設定
        self.user_key = user_key
        self.assistant_key = assistant_key

        #各エントリが「テキストのみサンプルか」をあらかじめリスト化（バッチ組成で利用可）
        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

    #長さ（サンプル数） を返します。
    def __len__(self) -> int:
        return len(self.list_data_dict)

    #サンプル取得
    def __getitem__(self, i) -> Dict[str, List]:      
        source = self.list_data_dict[i]

        #画像の読み込み
        images = []
        if "image" in source: #if "image" in source: で、そのサンプルに画像情報が含まれるかどうかを判定。
            # here we do not do any image preprocessing but rather
            # let the processor handle everything
            # in some cases this may cause slight differences
            # but should totally be fine (e.g., official llava-1.5 does padding,
            # but llava-1.5-hf (huggingface's implementation) does not)

            #JSON では "image": "img.jpg" のように単一文字列か、"image": ["a.jpg","b.jpg"] のようにリストか、実装依存で混在します。
            #isinstance(..., list)／isinstance(..., str) を使って、どちらでも必ず image_sources がリストになるように正規化しています。
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")
            
            #JSON 中では "image": "img1.jpg" のようにファイル名だけが書かれていることが多いので、self.image_folder（例："./example_data/images"）が指定されていれば、
            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)#"./example_data/images/img1.jpg" のようにフルパスに変換します
                #モデルに合わせた対応
                images.append(
                    Image.open(image_path).convert("RGB") #RGB モードの PIL.Image オブジェクトに変換して返します。
                    if self.load_image else image_path #のままパス文字列を返し、後続の processor／data_collator で読み込むようにします。
                )

        #動画の読み込み
        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")

            num_frames = [self.num_frames] * len(video_sources)

            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)
        
        #会話データと system_prompt の整形
        #オプションの system_prompt を取り出し。
        #会話リスト (conversations) が ユーザー→アシスタント→… の順で交互に並んでいるかをチェックしつつ、
        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        #各ターンの発話文 (value) だけを順に convs リストに詰める。
        convs = []
        assert len(source["conversations"]) > 0, "No conversations found" #長さチェック: 会話が１ターン以上あるか否かを assert で確認。
        #役割チェック: i が偶数なら "from" がユーザーキー（デフォルト "human"）、奇数ならモデルキー（デフォルト "gpt"）になっているかを検証。
        for i, conv in enumerate(source["conversations"]):
            #conv["from"] 偶数インデックスなら user_key、奇数なら assistant_key と一致するかチェック
            # conversations は必ず
            # ユーザー → モデル → ユーザー → モデル → … 
            # という順番で並んでいなければいけません。そこでこのチェックを入れています。
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), "Invalid conversation"
            #発話内容 ("value") のみを順に convs リストに詰める。
            convs.append(conv["value"])
            
        #偶数要素数チェック: 最終的にユーザー→モデル→ユーザー→モデル…とペアになるよう、要素数が偶数かどうかを確認。
        assert len(convs) % 2 == 0, "Odd number of conversations"
        
        #辞書として返却
        return dict(
            images=images, # 画像のリスト（PIL.Image or path）
            videos=videos, # 動画フレームの np.ndarray リスト
            conversations=convs, # 発話文のみのリスト
            system_prompt=system_prompt
        )
