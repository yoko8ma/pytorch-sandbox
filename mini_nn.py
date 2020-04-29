import torch
import torch.nn as nn


INPUT_FEATURES = 2  # 入力（特徴）の数
OUTPUT_NEURONS = 1  # ニューロンの数

activation = torch.nn.Tanh()    # 活性化関数： tanh関数

# torch.nn.Moduleクラスのサブクラス化： 典型的な書き方なので、最もお勧めできる
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 層（layer：レイヤー）を定義
        self.layer1 = nn.Linear(    # Linearは「全結合層」を指す
            INPUT_FEATURES,     # データ（特徴）の入力ユニット数
            OUTPUT_NEURONS,     # 出力結果への出力ユニット数
        )

    # forward関数： フォワードパス（＝活性化関数で変換しながらデータを流す処理）を実装する
    def forward(self, input):
        output = activation(self.layer1(input))     # 活性化関数は変数として定義
        # 「出力＝活性化関数（第n層（入力））」の形式で記述する。
        # 層（layer）を重ねる場合は、同様の記述を続ければよい
        # 「出力（output）」は次の層（layer）への「入力（input）」に使う。
        # 慣例では入力も出力も「x」と同じ変数名で記述する
        return output


model = NeuralNetwork()
print(model)


# パラメーター（ニューロンへの入力で必要となるもの）の定義
weight_array = nn.Parameter(
    torch.tensor([[ 0.6,
                   -0.2]]))  # 重み
bias_array = nn.Parameter(
    torch.tensor([  0.8 ]))  # バイアス

# 重みとバイアスの初期値設定
model.layer1.weight = weight_array
model.layer1.bias = bias_array

# torch.nn.Module全体の状態を辞書形式で取得
params = model.state_dict()
#params = list(model.parameters()) # このように取得することも可能
print(params)
# 出力例：
# OrderedDict([('layer1.weight', tensor([[ 0.6000, -0.2000]])),
#              ('layer1.bias', tensor([0.8000]))])


X_data = torch.tensor([[1.0, 2.0]])  # 入力する座標データ（1.0、2.0）
print(X_data)
# tensor([[1., 2.]]) ……などと表示される

y_pred = model(X_data)  # このモデルに、データを入力して、出力を得る（＝予測：predict）
print(y_pred)
# tensor([[0.7616]], grad_fn=<TanhBackward>) ……などと表示される


from torchviz import make_dot  # 「torchviz」モジュールから「make_dot」関数をインポート
make_dot(y_pred, params=dict(model.named_parameters()))
# 引数「params」には、全パラメーターの「名前: テンソル」の辞書を指定する。
# 「dict(model.named_parameters())」はその辞書を取得している


x = torch.tensor(1.0, requires_grad=True)  # 今回は入力に勾配（gradient）を必要とする
# 「requires_grad」が「True」（デフォルト：False）の場合、
# torch.autogradが入力テンソルに関するパラメーター操作（勾配）を記録するようになる

#x.requires_grad_(True)  # 「requires_grad_()」メソッドで後から変更することも可能

y = x ** 2     # 「yイコールxの二乗」という計算式の計算グラフを構築
print(y)       # tensor(1., grad_fn=<PowBackward0>) ……などと表示される

y.backward()   # 逆伝播の処理として、上記式から微分係数（＝勾配）を計算（自動微分：Autograd）

g = x.grad     # 与えられた入力（x）によって計算された勾配の値（grad）を取得
print(g)       # tensor(2.)  ……などと表示される
# 計算式の微分係数（＝勾配）を計算するための導関数は「dy/dx=2x」なので、
#「x=1.0」地点の勾配（＝接線の傾き）は「2.0」となり、出力結果は正しい。
# 例えば「x=0.0」地点の勾配は「0.0」、「x=10.0」地点の勾配は「20.0」である


# 勾配計算の前に、各パラメーター（重みやバイアス）の勾配の値（grad）をリセットしておく
model.layer1.weight.grad = None      # 重み
model.layer1.bias.grad = None        # バイアス
#model.zero_grad()                   # これを呼び出しても上記と同じくリセットされる

X_data = torch.tensor([[1.0, 2.0]])  # 入力データ（※再掲）
y_pred = model(X_data)               # 出力結果（※再掲）
y_true = torch.tensor([[1.0]])       # 正解ラベル

criterion = nn.MSELoss()             # 誤差からの損失を測る「基準」＝損失関数
loss = criterion(y_pred, y_true)     # 誤差（出力結果と正解ラベルの差）から損失を取得
loss.backward()   # 逆伝播の処理として、勾配を計算（自動微分：Autograd）

# 勾配の値（grad）は、各パラメーター（重みやバイアス）から取得できる
print(model.layer1.weight.grad) # tensor([[-0.2002, -0.4005]])  ……などと表示される
print(model.layer1.bias.grad)   # tensor([-0.2002])  ……などと表示される
# ※パラメーターは「list(model.parameters())」で取得することも可能