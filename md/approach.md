## アプローチ
- unet resnet34
- classification with resnet34, seresnext101, inceptionresnet
- 分類でふるいにかける→セグメンテーション
- unetのencoderを変える×→lrが高いのかも

## モデル追加要素
- decoder skip connect○
- encoder cbam○
- decoder cbam○
- freeze bn○
- hcolumns△
- aspp横方向のみ○
- decoder swish△だけど使ってる
- last conv3✖︎3×
- unetのbranch切ってclassification ○
- cutmix×→ふつうのcutmix以外にも、縦にばさっと切ってくっつけるとかも試した
- ema△→lossは下がるけど、scoreが上がらない、、、

## 学習
- bce○
- bce dice×
- bce focal dice×
- lovasz×
- bce→lovaszたぶん○

## データ周り
- ガンマ補正○？
- 輝度の標準化×？
- sample rate○

## その他
- ずっとlossが最小のモデルを保存してきたが、lossが最小≠metricが最大 のようである。そこで、今日metric最大を保存するとかしてみたが、スコア下がったのでよくわからない。
- データの明るさがバラバラなので、ガンマ補正とかしてて、lossは0.8切るくらいから0.77切るくらいまで下がったけど、スコアは上がってない
- class2は全然当たらないので全部nanが1番良い。discussionで上位の人もそれ言ってた。

## discussion情報で使えてないもの
- 画像がいろんなフレームから来ていて、それをclusterにまとめられる。それごとにtargetの特徴があり、後処理で結構使えるらしい。
- あとbinaryじゃなくsoftmaxでやる方法も出てて、試してない。

## あと試したいこと
- pseudo labeling
