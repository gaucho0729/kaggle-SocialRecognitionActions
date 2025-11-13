# データセットの説明

マウスの行動を注釈付ける作業は、現在非常に時間のかかる作業です。このコンテストでは、このプロセスを自動化し、動物行動に関する新たな知見を解き明かすことを目指します。

本コンテストでは非公開のテストセットを使用します。提出されたノートブックが採点されると、実際のテストデータ（完全なサンプル提出データを含む）がノートブックで利用可能になります。非公開テストセットには約200本の動画が含まれることを想定してください。

## ファイル

* [train/test].csv マウスと記録設定に関するメタデータ。

    lab_id - データ提供研究室の仮称。CalMS21、CRIM13、MABe22データセットは追加訓練データとして提供された公開データセットの複製です。CalMS21セット内の追跡ファイルの一部は重複しています。これらは異なる行動セットについて複数の人物によって注釈が付けられました。
    video_id - 動画の一意の識別子。
    mouse[1-4] [系統/毛色/性別/ID/年齢/条件] - 各マウスに関する基本情報。 (object/float64)
    frames_per_second - フレームレート (fps)
    video_duration_sec - 動画再生時間 (秒)
    pix_per_cm_approx - ピクセル/cm (概算)
    video_width_pix / video_height_pix - 動画 [幅/高さ]
    arena_width_cm / arena_height_cm - アリーナ [幅/高さ] (cm)
    arena_shape - アリーナ形状 (object)
    arena_type - アリーナタイプ (object)
    body_parts_tracked - 各研究室でマウス追跡技術が異なるため、追跡部位は異なる場合があります。(object)
    behaviors_labeled - この動画でラベル付けされた行動。アノテーションが疎なため必要です。各研究室で動画のアノテーション方法が異なり、動画内の全行動や全マウスがアノテーションされていない場合があります。(object)
    tracking_method - 動物の姿勢を追跡するために使用されたモデル。

* [train/test]_tracking/ 特徴データ。

    video_frame
    mouse_id
    bodypart - 追跡された身体部位。研究室によって追跡部位が異なる場合あり
    [x/y] - 身体部位のX/Y座標位置（ピクセル単位）。

* train_annotation/ トレーニング用ラベル。

    agent_id - 行動を実行しているマウスのID。
    target_id - 行動の対象となっているマウスのID。自己グルーミングなど一部の行動では、エージェントIDとターゲットIDが同一となる。
    action - 発生している行動の種類（例：グルーミング、追いかけ）。研究室によって注釈付けられた行動は異なる。
    [start/stop]_frame - 動作の最初/最後のフレーム。

* sample_submission.csv 正しい形式の提出用ファイル。

    row_id - 一意の行ID列を提供する必要があります。単純な列挙で十分です。
    video_id
    agent_id
    target_id
    action
    [start/stop]_frame

