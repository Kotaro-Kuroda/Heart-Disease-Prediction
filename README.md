# Heart Disease Prediction

## データセット
[ここ](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)のデータセットを用いて患者が心臓病であるかどうかを判定するモデルを学習する。

予測に用いる特徴量は以下のようになっている。

|変数名|説明|
|--|--|
|Age|患者の年齢 \[years\]|
|Sex|患者の性別 \[M: 男性, F: 女性\]|
|ChestPainType|胸痛の種類 \[TA: 典型的狭心症、ATA: 非典型的狭心症、NAP:非狭心症性疼痛、ASY: 無症状\]|
|RestingBP|安静時血圧 \[mmHg\]|
|Cholesterol| 血清コレステロール \[mm/dl\]|
|FastingBS|空腹時血糖値 \[1: FastingBS > 120 mg/dl, 0: それ以外\].|
|RestingECG| 安静時心電図結果 \[Normal:正常、ST：ST-T波異常（T波反転及び／又は0.05mV以上のST上昇又はST低下）、LVH：Estesの基準で左心室肥大の可能性が高い、または明確な左心室肥大を示すもの\] |
|MaxHR| 最大心拍数 \[60~202\]|
|ExerciseAngina| 運動誘発性狭心症 \[Y：Yes, N：No\]|
|Oldpeak| 安静時と比較して運動により誘発されたST低下|
|ST_Slope| ピークエクササイズSTセグメントの傾斜 \[Up: 上り傾斜、Flat: 平坦、Down: 下り傾斜\] |
|HeartDisease| 出力クラス \[1：心臓病、0：正常\]|

データセットは以下のようになっている（先頭9行のみ表示）。
| Age  | Sex  | ChestPainType | RestingBP | Cholesterol | FastingBS | RestingECG | MaxHR | ExerciseAngina | Oldpeak | ST_Slope | HeartDisease |
| :--- | :--- | :------------ | :-------- | :---------- | :-------- | :--------- | :---- | :------------- | :------ | :------- | :----------- |
| 40   | M    | ATA           | 140       | 289         | 0         | Normal     | 172   | N              | 0       | Up       | 0            |
| 49   | F    | NAP           | 160       | 180         | 0         | Normal     | 156   | N              | 1       | Flat     | 1            |
| 37   | M    | ATA           | 130       | 283         | 0         | ST         | 98    | N              | 0       | Up       | 0            |
| 48   | F    | ASY           | 138       | 214         | 0         | Normal     | 108   | Y              | 1.5     | Flat     | 1            |
| 54   | M    | NAP           | 150       | 195         | 0         | Normal     | 122   | N              | 0       | Up       | 0            |
| 39   | M    | NAP           | 120       | 339         | 0         | Normal     | 170   | N              | 0       | Up       | 0            |
| 45   | F    | ATA           | 130       | 237         | 0         | Normal     | 170   | N              | 0       | Up       | 0            |
| 54   | M    | ATA           | 110       | 208         | 0         | Normal     | 142   | N              | 0       | Up       | 0            |

## 結果

|モデル|正答率|
|--|---|
|Neural Network|0.858|
|Support Vector Machine|0.853|