# [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction)

## Description
ボラティリティは、トレーディングフロアで最もよく耳にする言葉の一つですが、それには理由があります。金融市場では、ボラティリティーは価格の変動量を表します。ボラティリティが高いということは、市場が乱高下し、価格が大きく変動していることを意味し、一方、ボラティリティが低いということは、より穏やかで落ち着いた市場を意味します。オプティバーのような取引会社にとって、ボラティリティを正確に予測することは、価格が原商品のボラティリティに直接関係するオプションの取引に不可欠です。

オプティバーは、世界有数の電子マーケットメーカーとして、金融市場の継続的な改善に取り組んでおり、世界中の数多くの取引所で、オプション、ETF、現物株式、債券、外国通貨へのアクセスと価格の向上を実現しています。オプティバーのチームは、数え切れないほどの時間をかけて、ボラティリティーを予測し、最終投資家にとってより公平なオプション価格を継続的に生成する高度なモデルを構築してきました。しかし、業界をリードする価格決定アルゴリズムは進化を止めることはできません。オプティバーがそのモデルを次のレベルに引き上げるために、Kaggleほど最適な場所はありません。

このコンペティションでは、最初の3ヶ月間で、様々なセクターの数百銘柄の短期的なボラティリティを予測するモデルを構築していただきます。何億行もの非常に詳細な財務データを手に入れ、それをもとに10分間のボラティリティーを予測するモデルを設計します。作成したモデルは、トレーニング後の3ヶ月間の評価期間に収集した実際の市場データと比較して評価されます。

このコンペティションを通じて、皆さんはボラティリティーと金融市場の構造に関する貴重な洞察を得ることができます。また、オプティバーが何十年にもわたって直面してきたデータサイエンスの問題についても、より深く理解することができるでしょう。私たちは、Kaggleコミュニティがこの複雑でエキサイティングなトレーディング課題にクリエイティブなアプローチを適用することを楽しみにしています。

はじめに

オプティバーのデータサイエンティストは、Kaggle参加者がこのコンペティションに向けてより良い準備ができるように、コンペティションのデータやこのトレーディング・チャレンジに関連する金融コンセプトについての報告をまとめたチュートリアル・ノートブックを作成しました。また、オプティバーのオンラインコースでは、金融市場やマーケットメイキングについて詳しく説明しています。


## Data
このデータセットには、金融市場での実際の取引の実行に関連する株式市場データが含まれています。特に、オーダーブックのスナップショットと約定した取引が含まれています。1秒の分解能で、現代の金融市場のミクロ構造を独特のきめ細かさで見ることができます。

これは、テストセットの最初の数行のみがダウンロード可能なコードコンペです。表示される行は、隠れたテストセットのフォーマットとフォルダ構造を説明するためのものです。残りの行は、提出されたノートブックでのみ利用できます。隠しテストセットには、約150,000のターゲット値を予測するための特徴量を構築するために使用できるデータが含まれています。データセット全体の読み込みには、推定で3GBを若干超えるメモリが必要になります。

本大会は予測競技でもあり、最終的なプライベートリーダーボードはトレーニング期間終了後に収集されたデータを用いて決定されるため、パブリックリーダーボードとプライベートリーダーボードの重なりはゼロとなります。これは、隠しデータセットのサイズが実際のテストデータとほぼ同じになるようにするためです。予測段階では、フィラーデータは完全に削除され、実際の市場データに置き換えられます。

**ファイル名**

**book_[train/test].parquet**

stock_id で分割された parquet ファイル。市場に入力された最も競争力のある買い注文と売り注文のオーダーブックデータを提供します。ブックの上位2階層は共有されます。ブックの最初のレベルは価格面でより競争力があるため、2番目のレベルよりも優先的に執行されることになります。

* stock_id - 銘柄のIDコード。すべての銘柄IDがすべての時間バケットに存在するわけではありません。Parquetはロード時にこのカラムをカテゴリカルデータタイプに変換しますので、int8に変換するとよいでしょう。
* time_id - 時間バケットのIDコードです。時間IDは必ずしも連続したものではなく、すべての銘柄で一貫しています。
* seconds_in_bucket - バケットの開始からの秒数で、常に0から始まります。
* bid_price[1/2] - 最も競争力のある買いレベル/2番目に競争力のある買いレベルの正規化された価格。
* ask_price[1/2] - 最も競争力のある売りレベル/2番目に競争力のある売りレベルの正規化された価格です。
* bid_size[1/2] - 最も/2番目に競争力のある買いレベルの株式数。
* ask_size[1/2] - 最も競争力のある/2番目に競争力のある売りレベルの株式数。

**trade_[train/test].parquet**

stock_id で分割された parquet ファイル。実際に実行されたトレードのデータが含まれています。通常、市場では実際の取引よりも受動的な売買意思の更新（ブックアップデート）が多いため、このファイルはオーダーブックよりもまばらであることが予想されます。

* stock_id - 上記と同じです。
* time_id - 上記と同じです。
* seconds_in_bucket - 上記に同じ。取引データとブックデータは同じ時間軸で取得されており、一般的に取引データの方がまばらであるため、このフィールドは必ずしも0から始まるわけではないことに注意してください。
* price - 1秒間に実行された取引の平均価格。価格は正規化されており、平均値は各取引で取引された株式数で加重されています。
* size - 取引された株式数の合計です。
* order_count - 行われたユニークな取引注文の数。

**train.csv** 

トレーニングセットのグランドトゥルース値です。

* stock_id - 上記と同じですが、csvであるため、カラムはカテゴライズされずに整数としてロードされます。
* time_id - 上記と同じです。
* target - 同じ銘柄/time_idの特徴データに続く10分間のウィンドウで計算された実現ボラティリティ。特徴データとターゲットデータの間に重複はありません。詳細はチュートリアルノートを参照してください。

**test.csv** 

他のデータファイルと投稿ファイルの間のマッピングを提供します。他のテストファイルと同様に、データのほとんどは送信時にノートブックでのみ利用でき、最初の数行だけがダウンロード可能です。

* stock_id - 上記と同じです。
* time_id - 上記と同じです。
* row_id - 提出された行のユニークな識別子です。既存の時刻ID/銘柄IDのペアごとに1つの行があります。各時間帯には必ずしも個々の銘柄が含まれているわけではありません。

**sample_submission.csv** 

正しいフォーマットのサンプル投稿ファイルです。

* row_id - test.csvの定義と同じです。
* target - train.csvと同じ定義です。ベンチマークではtrain.csvの中央値のターゲット値を使用しています。


## Score

[notion](https://www.notion.so/Score-8b319ffc71294c76a0182a3e784363b2)にまとめておく

---

## Log

### 20210817

* 今日から日記つける
* やってたこと
    * lgbm-startをとりあえず動かしてみた
    * その他Discussion, notebookを読んだ
* ## [金融について説明](https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data)

  - オーダーブック

    ![orderbook1](https://www.optiver.com/wp-content/uploads/2021/05/OrderBook3.png)

    各価格帯で買い付けられている株数や売り出されている株数が記載されている。

    - 一般的なオーダーブックの統計
        - bid/ask spread
        - 加重平均価格

            WAP = BidPrice1∗AskSize1 + AskPrice1∗BidSize1 / (BidSize1+AskSize1)

  - マーケットメーカー

    ![orderbook2](https://www.optiver.com/wp-content/uploads/2021/05/OrderBook5.png)

    非効率的な市場に流動性を持たせる人たちのこと。

    買い注文と売り注文の両方を表示し、流動性を持たせることで効率的な市場を作る。

  - ログリターン

    リターンを表すとき割合のlogを取って表すことが多い。

    足し算で連続する時間のリターンを表せる。

  - realized volatility

    オーダーブックの全アップデートごとのログリターンを計算しその２乗和の平方根でvolatilityを定義。
    
    与えられた最初の10分間のvolatilityをreallzed volatilityとして特徴量に使っている？

    ログリターンの計算のために加重平均価格を使用している。

* nb001
  * lightgbmでとりあえず提出
  * score: 0.21937
* pickleで特徴量とモデルを保存してnotebookで読み込み
  * pythonのversionを合わせてなかったからエラー出た。


### 20210818
* nb002
  * pickleで特徴量とモデル保存
* nb003
  * 特徴量とモデルを読み込んでテストするテンプレ
* 銘柄間の相関関係をモデルに組み込めればうまくいくかも？[Discussion](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/256356])

### 20210819
* pd.read_pickleがkaggleのノートブックでうまく動かない。-> pickle.dump()で保存したらうまく行った。
* nb004
  * stock_id, time_idごとに集約し統計量をとったものを特徴量に追加
* [AutoEncoderで特徴量を追加してMLPで予測](https://www.kaggle.com/gogo827jz/jane-street-supervised-autoencoder-mlp?scriptVersionId=56093968)
* 予測用のノートをkaggleの方で作ったけど提出したらスコアが下がりすぎておかしい。ちゃんと読み込めてるから単純に学習がおかしい？->予測に使う特徴量を指定していなかったから（なんで指定しないといけないのかよくわからん）lgbmわかってないな

### 20210820
* 昨日のlgbmの件
  * trainとtestで特徴量の順番が違うため揃える必要があった！！
* idごとに別のモデルで学習させる
* [metricとfevalについて](https://tawara.hatenablog.com/entry/2020/05/09/162633)
* nb006
  * [bookデータで落ちている秒数の部分をffillで埋める](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/251277)
* nb005
  * nb004の特徴量で学習
* nb007
  * ffillしたデータで学習したモデル作成
  * score down

### 20210821
* nb008
  * ffillなしnb005と同じ結果になるのか確認
* [以前のマーケット関連のコンペ](https://www.kaggle.com/c/jane-street-market-prediction/)
* nb009
  * issue9, 10を実験
  * https://www.kaggle.com/junjitakeshima/optiver-beginner-s-gradual-improvement-eng
* nb010
  * lgbm gpuで学習
  * targetにlogだけ適用

### 20210824
* 3日も空けてしまった・・・毎日何かしらの成果を残す！
* targetをlogとって実験したが、ValScoreが下がった（重み付けのせいか？）
* lgbm gpuがかなり遅くなった。。。なぜ？

### 20210828
* autoencoderで特徴量作ってみる
* dataset, dataloader

### 20210829
* nb012
  * denoising autoencoderを実装、noiseを入れる処理がなかなか時間かかる。。。
  * ミニバッチ学習の際のvalidationについて質問させていただいた。

### 20210830
 * denoising autoencoderで圧縮したデータを追加で入れてみてlgbmしてみる
 * output/feature_model/20210830/1 ... DAEのモデル
 * lgbm-gpugpuほぼつかってないし、cpuの方に速さがかなり依存してるっぽい？誰も使ってないサーバで動かしたら早かった。
 * 新たな特徴量の参考→ https://www.kaggle.com/hijest/lgbm-nn-ensamble-2-0
 * output/feature_model/20210830/3 ...現状一番良い

### 20210831
 * [ARCHモデル](https://ja.wikipedia.org/wiki/ARCH%E3%83%A2%E3%83%87%E3%83%AB)
 * 参考をもとに特徴量追加→output/feature_model/20210831/1
 * DAEで特徴量作成→output/feature_model/20210831/4

### 20210901
 * [discussion](https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/266354) シンプルなモデルがよい？
 * stock_idを特徴量に入れると過学習するという意見もある。
 * 2:DAEの特徴量追加
 * new_feature_lgbm2:　tauなどの特徴量
 * embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
   * 今回はstock_idで埋め込みを行う

### 20210901
* kerasのcodeをpytorchに移植

### 20210902
* ひとまず思いつく特徴量をひたすら作る
* ニューラルネットの初期値考える
* t-sneで次元圧縮した特徴量追加
  * 3次元以上は時間がかかる
  * 元の特徴量が大きいと時間かかるし正確に行かないかも
  * pcaしたものとかを入れるといい？
* pcaで圧縮したデータを特徴量に追加

### 20210903
* epochごとにgpu使用メモリが増えてってエラーになる。。。（FFNN）
* DataSet, DataLoaderあたりが怪しい？
  * Dataloaderで返された値を随時gpuに渡す方が良さそう。
* いや、lossを保存するときにitem()つけ忘れてtensorのまま保存してたからじゃないか？
  * 計算グラフ全部持っとかないといけないからメモリだんだん食われて行ったのでは？→そうだったみたい。。。めっちゃハマった。。。
* volumeで割った値を特徴量追加
* output/feature_model/20210904/lgbm_add_feature: 割り算とかしてみた
  * cvが上がった。realized_volatility系がかなり重要そう
  * LBは上がらんかった

### 20210904
* クラスタリングして特徴量を作る


### 20210907
* time_idがstrになってた。intに変換して処理。
* time_idでgroupkfoldの必要あり？
* StratifiedGroupKfold使うべきでは？
* 単純なKfoldの方がいいって言ってる人がいた（https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/262185）
* GroupKFoldで行うと特定の時間帯の予測がかなり困難となってスコアが下がるみたい

### 20210908
* stock_id内の平均との差を特徴量に追加
* time_id内でも同様に作る？
* stratified k-fold試す
* test時clusteringの特徴量追加部分でエラー
* stock_idについて、intで保存したはずなのにobjectになっている。pickleでloadするとなるのか？-> そんなわけない。target encodingの際にobjectに変換されてる

### 20210909
* debug modeを追加（データ減らして動かす）

### 20210910
* 特徴量選択
  * 相関係数が高いものを削除
* ニューラルネットにdropoutとbatchnormは必ずしも効くとは限らない。
  * https://towardsdatascience.com/pitfalls-with-dropout-and-batchnorm-in-regression-problems-39e02ce08e4d
* memory削減の関数
* stock_id=31だけ圧倒的にスコアが悪い問題（https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/269282）
  * stock_id=31だけ別に学習するとか？


### 20210912
* pivotデータ型に注意しないといけないらしい

### 20210920
* とりあえずtime_idでgroup_k_fold
* timeの特徴量を消したほうがいい
* reduce_mem使うとスコアがかなり悪くなる。lgbmの計算に支障出てる？
* stock_id=31抜きで学習し、推論を行ったところかなり悪い結果となった。
> Performance of the　prediction: , RMSPE: 0.189
> 
> Performance of the　prediction stock_id=31: , RMSPE: 1.223

* 31は別で学習したほうがいいな（他にも別で学習したほうがいいやつありそうだけど）
* nb021: 31かどうかの特徴量を追加してみるか
  * cvスコア下がった
  * 別で学習したほうが良さげ
  * 別で学習して合体させる方法試してみたが、それほど変わらなかった。
* group_k_foldかなりスコア上がった。time_idでのgroup_k_foldのモデル使うべきな気がするーーー

### 20210921
> RuntimeError: CUDA error: device-side assert triggered
* cudaのエラー
  * modelの入力のサイズが違うから？消えないな
  * cudaのバージョンは変えたけどダメだった
  * .to(torch.device('cuda:0'))、cuda()どっちでも関係なかった
  * 1foldの時は大丈夫なのに2foldからダメになる

### 20210922
  * pytorchとcudaをもっかい設定→だめ
  * 前に動いてたのでもだめだった
  * gpu変えたら動いた（前のやつ）gpu01はだめだった　05でいけた
  * updateAEではだめだった
  * datasetがだめそう
  * 変数のミス！！後で同じ変数使ってた...
  * quadroだと動く


### 20210923
* autoencoderで圧縮した特徴量を追加してlgbmを回す
  * cvさがってる


### 20210924
* optunaでハイパラチューニング

### 20210925
* group, stratified を使うべきかどうかかなり意見が割れている。リークしているって意見もあるけどLBスコアが高くなっててわからん。
* output/feature_model/20210920/lgbm_group_kfold_time_id
* reshapeの際定数で指定したため、Notebook Threw Exception

### 20210926
* パラメータ変えたモデル色々学習→アンサンブル
* DAEを最初に学習してからだとCVでリークが起こる可能性がある。バリデーションで使うデータもDAE学習時使ってるから
* 
