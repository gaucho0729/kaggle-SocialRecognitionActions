train.csv:

    lab_id                 object : データ提供研究室の仮称。CalMS21、CRIM13、MABe22データセットは追加訓練データとして提供された公開データセットの複製です。CalMS21セット内の追跡ファイルの一部は重複しています。これらは異なる行動セットについて複数の人物によって注釈が付けられました。
    video_id                int64 : 動画の一意の識別子。
    mouse1_strain          object : マウス1 系統
    mouse1_color           object : マウス1 毛色
    mouse1_sex             object : マウス1 性別
    mouse1_id             float64 : マウス1 ID
    mouse1_age             object : マウス1 年齢
    mouse1_condition       object : マウス1 条件
    mouse2_strain          object : マウス2 系統
    mouse2_color           object : マウス2 毛色
    mouse2_sex             object : マウス2 性別
    mouse2_id             float64 : マウス2 ID
    mouse2_age             object : マウス2 年齢
    mouse2_condition       object : マウス2 条件
    mouse3_strain          object : マウス3 系統
    mouse3_color           object : マウス3 毛色
    mouse3_sex             object : マウス3 性別
    mouse3_id             float64 : マウス3 ID
    mouse3_age             object : マウス3 年齢
    mouse3_condition       object : マウス3 条件
    mouse4_strain          object : マウス4 系統
    mouse4_color           object : マウス4 毛色
    mouse4_sex             object : マウス4 性別
    mouse4_id             float64 : マウス4 ID
    mouse4_age             object : マウス4 年齢
    mouse4_condition       object : マウス4 条件
    frames_per_second     float64 : フレームレート
    video_duration_sec    float64 : ビデオ再生時間
    pix_per_cm_approx     float64 : ピクセル(cm)
    video_width_pix         int64 : 動画幅
    video_height_pix        int64 : 動画高さ
    arena_width_cm        float64 : アリーナ幅(cm)
    arena_height_cm       float64 : アリーナ高さ(cm)
    arena_shape            object : アリーナ形状
    arena_type             object : アリーナ種類
    body_parts_tracked     object : 追跡部位
    behaviors_labeled      object : ラベル付けされた行動
    tracking_method        object : 追跡モデル


test.csv:

    row_id      : 行ID
    video_id    : 動画の一意の識別子 (from train.csv)
    agent_id    : 行動を実行しているマウスのID (from train_annotation/*/*.parquet)
    target_id   : 行動の対象となっているマウスのID (from train_annotation/*/*.parquet)
    action      : 発生している行動の種類（例：グルーミング、追いかけ）
    start_frame : 動作の最初のフレーム
    stop_frame  : 動作の最後のフレーム






[train/test]_tracking/ 特徴データ:

    video_frame
    mouse_id
    bodypart - 追跡された身体部位。研究室によって追跡部位が異なる場合あり
    [x/y] - 身体部位のX/Y座標位置（ピクセル単位）。



train_annotation/ トレーニング用ラベル:

    agent_id - 行動を実行しているマウスのID。
    target_id - 行動の対象となっているマウスのID。自己グルーミングなど一部の行動では、エージェントIDとターゲットIDが同一となる。
    action - 発生している行動の種類（例：グルーミング、追いかけ）。研究室によって注釈付けられた行動は異なる。
    [start/stop]_frame - 動作の最初/最後のフレーム。


action:
    allogroom
    approach
    attack
    attemptmount
    avoid
    biteobject
    chase
    chaseattack
    climb
    defend
    dig
    disengage
    dominance
    dominancegroom
    dominancemount
    ejaculate
    escape
    exploreobject
    flinch
    follow
    freeze
    genitalgroom
    huddle
    intromit
    mount
    rear
    reciprocalsniff
    rest
    run
    selfgroom
    shepherd
    sniff
    sniffbody
    sniffface
    sniffgenital
    submit
    tussle

bodypart:
    body_center
    ear_left
    ear_right
    forepaw_left
    forepaw_right
    head
    headpiece_bottombackleft
    headpiece_bottombackright
    headpiece_bottomfrontleft
    headpiece_bottomfrontright
    headpiece_topbackleft
    headpiece_topbackright
    headpiece_topfrontleft
    headpiece_topfrontright
    hindpaw_left
    hindpaw_right
    hip_left
    hip_right
    lateral_left
    lateral_right
    neck
    nose
    spine_1
    spine_2
    tail_base
    tail_middle_1
    tail_middle_2
    tail_midpoint
    tail_tip
