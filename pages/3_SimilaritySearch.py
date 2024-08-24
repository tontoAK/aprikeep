import streamlit as st
import spacy
import pandas as pd

nlp=spacy.load('ja_ginza')
st.title("類似文検索")
#Input
input_text=st.text_input('検索')
#uploaded_file=st.file_uploader('CSV選択',type='csv')
uploaded_file = st.file_uploader("Excelを選択", type='xlsx')
#Process
if uploaded_file is not None:
  tg_data=pd.read_excel(uploaded_file)
  tg_col=st.selectbox('対象列選選択;テキスト以外はエラーになります',tg_data.columns)
  if tg_col is not None:
    if st.button('実行'):
      tg_data=tg_data.dropna()#欠損値削除
      tg_data.reset_index(drop=True,inplace=True)#インデックスふり直し
      tg_data['similarity']=0
      doc1=nlp(input_text)
      for i in range(len(tg_data)):
        doc2=nlp(tg_data[tg_col][i])
        similarity=doc1.similarity(doc2)#類似度計算
        tg_data['similarity'][i]=similarity
      tg_data.sort_values('similarity',ascending=False,inplace=True)
      tg_data.set_index(tg_col,inplace=True)
      
      #Output
      st.write('類似度の高い順に表示')
      st.dataframe(tg_data)
      #st.dataframe(tg_data[['similarity']])