# Dataset json file

```json
[
    {
        "system_prompt": "Answer the following questions about the image and video.",
        "video": ["bm_teaser.mp4", "bm_show.mp4"],
        "image": "bm.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<video><video>What are these videos about?"
            },
            {
                "from": "gpt",
                "value": "These videos are featuring a Kpop group, BabyMonster."
            },
            {
                "from": "human",
                "value": "<image>What does this image show?"
            },
            {
                "from": "gpt",
                "value": "This image shows the members of the Kpop group BabyMonster."
            }
        ]
    }
]
```

The above example shows one dataset entry that has all the keys that the code will look for (but some of them are not required to be presented all the time). Let's go over them one by one.

- `system_prompt`: This is the prompt that will be put at the very beginning of the conversation as a general instruction for the model. If there is no `system_prompt` key presented, then for the current sample there will simply be no system prompt.
- `video`: This is a list of paths (or a single string path) to the video(s). The paths could be relative or absolute, depending on whether the `video_folder` argument is specified to the training script. The only requirement is that the number of `<video>` token in the conversations should be the same as the number of videos in the current sample. If there is no `video` key presented, then the current sample will obviously have no video(s). Currently the script will sample a fixed number of frames from each video, and the number is specified by the `num_frames` argument to the training script. The reason for this (fixed number of frames) is that at the moment huggingface video models (e.g., LLaVA-NeXT-Video) do not unpad the video frames and do not have corresponding attention masks. So if we pad the video frames to the same length, the model will train on padded frames which is not ideal.
- `image`: This is a list of paths (or a single string path) to the image(s). The path could be relative or absolute, depending on whether the `image_folder` argument is specified to the training script. The only requirement is that the number of `<image>` token in the conversations should be the same as the number of images in the current sample. If there is no `image` key presented, then the current sample will obviously have no image(s).
- `conversations`: This is a list of conversation turns, alternating between the human/user and the model/assistant. Please make sure it strictly follows the order of human, model, human, model, human, model, ... Note, the role key is not fixed and can be specified by the `user_key` and `assistant_key` arguments to the training script. For instance, if your dataset uses "user" and "assistant" instead of "human" and "gpt", you can specify `user_key="user"` and `assistant_key="assistant"` to the training script. The `conversations` key is required to be presented in each dataset entry (otherwise there will be nothing to train on).


:warning: **If you have text-only entries in your training dataset**: the training is likely to fail at some point if 1) your `per_device_batch_size` is 1, or 2) the number of text-only instances dominate the number of multi-modal instances. This is due to a limitation/bug of deepspeed. If neither of the above two conditions is met, no worries, we got you covered.


以下の例は、コードが参照するすべてのキーを含むデータセットエントリを示しています（ただし、これらすべてが常に存在する必要はありません）。それぞれのキーについて、順に見ていきましょう。

system_prompt：モデルに対する一般的な指示として、会話の最初に配置されるプロンプトです。このキーが存在しない場合は、当該サンプルにシステムプロンプトは含まれません。

video：動画へのパス（または複数のパス）のリストです。パスは相対パスでも絶対パスでもかまいませんが、これは学習スクリプトに video_folder 引数を指定したかどうかによります。ここで注意すべきは、会話中の <video> トークンの数と、当該サンプルに含まれる動画ファイルの数が一致している必要がある、という点です。もし video キーが存在しなければ、そのサンプルには動画は含まれないことになります。現状、スクリプトは各動画から固定数のフレームをサンプリングするようになっており、その数は学習スクリプトに渡す num_frames 引数で指定します。このように固定数のフレームを使う理由は、執筆時点で Hugging Face の動画モデル（例：LLaVA-NeXT-Video）が動画フレームのパディングを解除せず、対応するアテンションマスクも持たないためです。その結果、動画フレームを同じ長さにパディングして学習させると、パディングされたフレームにまでモデルが学習してしまい望ましくない動作となるからです。

image：画像へのパス（または複数のパス）のリストです。パスは相対パスでも絶対パスでもかまいませんが、これは学習スクリプトに image_folder 引数を指定したかどうかによります。ここで注意すべきは、会話中の <image> トークンの数と、当該サンプルに含まれる画像ファイルの数が一致している必要がある、という点です。もし image キーが存在しなければ、そのサンプルには画像は含まれないことになります。

conversations：人間（ユーザー）とモデル（アシスタント）との会話ターンを交互に並べたリストです。必ず「人間→モデル→人間→モデル→…」という順序を厳守してください。なお、ロールを示すキー名（例：human や gpt）は固定ではなく、学習スクリプトの引数 user_key や assistant_key で指定できます。たとえば、データセット側で「user」「assistant」というキーを使っている場合は、学習スクリプトで user_key="user"、assistant_key="assistant" を指定すれば問題ありません。conversations キーは、各データセットエントリに必ず含まれている必要があります。これがないと学習に使う会話データが存在しないためです。

⚠️ トレーニングデータセットにテキストのみのエントリが含まれている場合：

per_device_batch_size を 1 に設定している、または
テキストのみのインスタンスがマルチモーダル（画像や動画を含む）インスタンスよりも多数を占めている(つまり、同様にマルチモーダル要素がゼロのバッチができやすくなる。)

という条件のいずれかが満たされると、DeepSpeed の制限／バグにより学習が途中で失敗する可能性があります。上記のいずれの条件にも該当しなければ、特に心配はいりません。

なぜなら、DeepSpeed は「各バッチに必ず画像や動画用のテンソルがある」ことを前提に内部のパディング・マスク処理をしているため、バッチがまるごとテキストのみになってしまうとエラーが起きやすいという制約があります。
つまり、DeepSpeed を使う場合はバッチ単位に画像などのテキスト以外のサンプルを混ぜないといけない、ということ。
