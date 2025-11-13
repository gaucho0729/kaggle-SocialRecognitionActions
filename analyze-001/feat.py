import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

all_data = pd.read_csv('train.csv')

if os.path.isdir('analyze')==False:
    os.makedirs('analyze')

all_data.info()
# 質的変数の列のみ抽出
qual_df = all_data.select_dtypes(include=['object', 'category'])

# 各質的変数のユニークな値の数を表示
unique_counts = qual_df.nunique()
print(unique_counts)
for col in qual_df.columns:
    print(f'【{col}】のカテゴリ一覧: {qual_df[col].unique()}')
    for group, sub_df in all_data.groupby(col):
        print(f'カテゴリ: {group}')
        corr = sub_df.select_dtypes(include='number').corr()
        fname = 'analyze/quant-'+col+'-'+group+'.csv'
        corr.to_csv(fname, index=False)
        print('{col}-{group}')


pd.set_option('display.max_columns', all_data.columns.size)
pd.set_option('display.max_rows', len(all_data))

# 量的変数のみを抽出
quant_df = all_data.select_dtypes(include='number')

# 相関係数行列を計算
correlation_matrix = quant_df.corr()

print(correlation_matrix)
correlation_matrix.to_csv('analyze/quant-corr-matrix.csv', index=False)

# 量的変数の散布図行列を作成する
# 相関係数行列のヒートマップを作成
def plot_corr(ax, data):
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")

# 散布図行列（ペアプロット）を作成
def plot_pairplot(data):
    return sns.pairplot(data, corner=True, plot_kws={"s": 10})

# ペアプロットは独立に作成し、後で組み合わせる
pairplot = plot_pairplot(quant_df)

# ペアプロットをSVGで一時保存し、画像として読み込む
pairplot.savefig("analyze/temp_pairplot.svg")

# 大きな1枚のキャンバスを作成（ヒートマップ＋ペアプロット）
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 2])

# ヒートマップ部分
ax0 = fig.add_subplot(gs[0])
plot_corr(ax0, quant_df)

# SVGのペアプロットを読み込んで右に表示（画像として）
import matplotlib.image as mpimg
import io

# SVG読み込み → PNG変換（matplotlibはSVGの直接貼り付けが難しいため）
import cairosvg
from PIL import Image

cairosvg.svg2png(url="analyze/temp_pairplot.svg", write_to="analyze/temp_pairplot.png")
img = mpimg.imread("analyze/temp_pairplot.png")

ax1 = fig.add_subplot(gs[1])
ax1.imshow(img)
ax1.axis('off')
ax1.set_title("Scatter Matrix")

# 最終的なSVGとして保存
plt.tight_layout()
plt.savefig("analyze/correlation_scatter_combined.svg", format="svg")
plt.show()
