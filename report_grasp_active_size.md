# 技術解説：AIによる多様なサイズの物体に対する適応的把持（アクティブ・サイズ）
**対象プログラム**: `grasp_active_size.py`

本稿では、サイズや位置が異なる物体（ボックス）に対して、強化学習（SAC）で訓練された5本指ハンド（Panda_HX5）がどのように適応し、確実な把持と持ち上げ（リフトアップ）を行うか、その推論プログラムの内部構造を解説します。
単なる固定動作の再生ではなく、物体の大きさに応じた動的な姿勢計算（IK）や、力覚フィードバックに基づく安定判定など、ロボットシミュレーション（Isaac Sim）における高度な制御メカニズムを紐解きます。

---

## 1. 対象物のサイズに応じた動的なプレシェイプ（事前姿勢）生成
対象物が大きければ指を大きく開き、小さければ狭める。人間が当たり前に行うこの「プレシェイプ（手を開く準備動作）」を、プログラム内で動的に計算しています。

```python
dx = self.current_box_size[0]
dy = self.current_box_size[1]
# ... 中略 ...
x_offset = max(0.0, (dx - 0.015) * 1.5)
y_offset = (dy - 0.04) * 1.5

self.base_finger_q[2::4] += (base_clearance + x_offset)
self.base_finger_q[1::4] += y_offset * 0.5
```
ボックスの寸法（`current_box_size`）から必要なクリアランス（`x_offset`, `y_offset`）を算出し、指の各関節の初期角度（`base_finger_q`）を決定しています。これにより、強化学習エージェントは「最初からある程度掴みやすい姿勢」から探索を開始でき、様々なサイズの物体に対して推論の成功率が飛躍的に向上します。

---

## 2. 逆運動学（IK）による正確なアプローチと位置補正
把持を成功させるには、ロボットの手首（TCP）を物体の真上に正確に配置する必要があります。このプログラムでは、物理シミュレータ本体（Isaac Sim）とは別に、軽量で高速な `PyBullet` をIKソルバとして裏側で併用しています。

```python
for it in range(5):
    # PyBulletでIKを計算
    q_ik = p.calculateInverseKinematics(...)
    
    # ... 中略（ロボットを動かして位置を確認） ...
    
    error = target_midpoint - current_midpoint
    err_norm = np.linalg.norm(error)
    
    if err_norm < 0.002: # 誤差が2mm未満ならOK
        break
        
    pb_target_midpoint = pb_target_midpoint + error * 0.8
```
目標とする把持中心点と、実際の親指・中指の中間点（`current_midpoint`）の誤差を計算し、許容誤差（0.002 = 2mm）に収まるまで目標位置を少しずつずらしながらIK計算を反復（Iterative IK）しています。これにより、複雑な多自由度ロボットアームでも精度の高いアプローチが可能になります。

---

## 3. 力覚と幾何情報による「安定把持」の検知
対象物を持ち上げる（リフト）タイミングは、単に「一定時間が経過したから」ではなく、ロボットが「確実に掴んだ」と判断した瞬間に行われます。

```python
# total_finger_effort: 指の関節にかかるトルクの合計
# avg_finger_dist: 指先から物体までの平均距離

is_force_balanced = (total_finger_effort > 0.45) and (avg_finger_dist < 0.09) and (self.step_counter > 15)

if is_force_balanced and not self.is_ready_to_lift:
    self.grasp_stable_count += 1
# ...
if self.grasp_stable_count >= 5 and not self.is_ready_to_lift:
    self.is_ready_to_lift = True # 持ち上げ動作に移行
```
ここでは以下の2つの条件を組み合わせています：
* **力覚 (`total_finger_effort > 0.45`)**: 指の関節にかかるトルク（力）の合計値。指が物体に接触し、対象物を押し付けていることを示します。
* **視覚/幾何 (`avg_finger_dist < 0.09`)**: 指先から物体中心までの平均距離。指が空振りせず、物体の周囲に適切に配置されているかを示します。

これらの条件が5ステップ連続で満たされた場合にのみ持ち上げ動作に移行します。これは、人間が手で物を掴むときの「手応え」を模倣したアルゴリズムと言えます。

---

## 4. グローバルキャッシュによる推論の高速化
実運用では、上位のAIから物体の位置・サイズを受け取り、何度も推論を回すことになります。そのたびにシミュレータ（Isaac Sim）を起動したり、数GBのモデルをロードしたりすると膨大な時間がかかります。

```python
global _active_app, _active_env, _active_model

if _active_app is None:
    _active_app = SimulationApp({"headless": headless})
# ...
if _active_model is None:
    _active_model = SAC.load(MODEL_PATH, env=vec_env, device="cuda:0")
```
`_active_app` や `_active_model` などをグローバル変数としてキャッシュ（保持）するシングルトンパターンを採用しています。これにより、2回目以降の `grasp_active_size()` 呼び出しでは重い初期化プロセスをスキップし、高速な連続推論（推論APIとしての機能）を実現しています。

---

## 5. まとめ
この推論プログラム `grasp_active_size.py` では、強化学習AI（SAC）がその真価を発揮できるように、周辺の制御技術がしっかりと脇を固めています。

* 物体サイズに応じた **プレシェイプ（事前姿勢）の動的生成**
* PyBulletを用いた **反復的な逆運動学（IK）による高精度な位置決め**
* トルクと距離の複合センシングに基づく **安定把持の検知**
* 連続推論システムに耐えうる **シミュレータとモデルのキャッシング**

純粋なAI技術だけでなく、古典的なロボット制御（IK）やヒューリスティックな判定（力覚検知）を組み合わせることで、多様な状況にロバスト（頑健）に対応できる実践的なロボットハンドのシステムが構築されています。
