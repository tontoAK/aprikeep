#初期定義
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import load
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import spacy

st.title("ポジネガ分析")
# input
# シリアライズされたモデルをロード
model = load('model_posinega_logi.pkl')
#データ読込
uploaded_file = st.file_uploader("コンビニおにぎり.xlsxを選択", type='xlsx')
if uploaded_file is not None:
  df_pred= pd.read_excel(uploaded_file)
  tg_col = st.selectbox('対象列選択,テキスト以外はエラーになります',df_pred.columns)

  #process
  if st.button('実行'):
    #文章ベクトルに変換して、xの値（特徴量）とする
    nlp = spacy.load('ja_ginza') # 学習済みモデルとしてginzaをロード
    vectors = []
    # (1) 1データごとに文章ベクトルに変換していく
    for _, sentence in df_pred[tg_col].items():
        doc = nlp(sentence)
        vectors.append(doc.vector)
    # (2) 文章ベクトルを説明変数に、PosiNegaカラムを目的変数にする
    X= vectors

    #予測
    result = model.predict(X)
    pred= np.where(result > 0, 'Positive', 'Negative')
    #pred= np.where(result > 0, 'Positive', np.where(result==0,'None','Negative'))
    df_pred['posinega_pred']=pred

    #円グラフ
    sizes = df_pred['posinega_pred'].value_counts() # まとまったカテゴリーの要素
    labels = df_pred['posinega_pred'].value_counts().index # カテゴリーネーム
    plt.figure(figsize=(2, 2)) # グラフ自体のサイズ（幅、高さ）
    textprops = {'fontsize':10}
    plt.pie(
      sizes,
      labels=labels,
      labeldistance=1.1,
      counterclock=True,
      startangle=90,
      pctdistance=0.7,
      autopct='%.1f%%', # 比率の表示
      wedgeprops=dict(width=0.8, edgecolor='w'), # widthで半径のサイズ、edgecolorで境界線の色指定
      shadow=False, # 円の影
      colors=["y","c"],
      textprops=textprops
    )
    plt.title('PosiNega', fontsize=14) # グラフタイトル
    #ポジネガ集計
    df_piv=pd.pivot_table(df_pred, index='datetime', columns='posinega_pred',values='no',aggfunc='count').fillna(0)
    
    #output
    st.write(f""" ### PosiNega Analysis""")
    st.dataframe(df_pred)
    #グラフ表示
    col1,col2=st.columns(2)
    with col1:
      st.write('ポジネガ集計')
      st.write(sizes)
    with col2:
      st.pyplot(plt)
    #折れ線グラフ
    st.write('ポジネガ推移:')
    st.line_chart(df_piv.set_index(df_piv.index))
    st.write('ポジネガ日付別集計:')
    st.dataframe(df_piv)