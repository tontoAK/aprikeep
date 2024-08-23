#初期設定
import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
from spacy.lang.ru.examples import sentences
from wordcloud import WordCloud
font_path = 'usr/share/fonts/opentype/ipafon-gothic/ipagp.ttf'

st.title("word cloud")

nlp = spacy.load('ja_ginza')
pos_dic = {'名詞':'NOUN', '代名詞':'PRON', 
           '固有名詞':'PROPN','動詞':'VERB','形容詞':'ADJ'}

# Input
#uploaded_file = st.file_uploader("CSVを選択", type='csv')
uploaded_file = st.file_uploader("Excelを選択", type='xlsx')
select_pos = st.sidebar.multiselect('品詞選択',
                ['名詞','代名詞','固有名詞','動詞','形容詞'],
                ['名詞'])

# Process
if uploaded_file is not None:
  #data = pd.read_csv(uploaded_file)
  data= pd.read_excel(uploaded_file)
  tg_col = st.selectbox('対象列選択,テキスト以外または容量制限でエラーになります',data.columns)

  if tg_col is not None:
    #include_pos = ('NOUN', 'PROPN', 'VERB', 'ADJ')
    tg_pos = [pos_dic[x] for x in select_pos]
    include_pos = tg_pos
    stopwords = ('する', 'ある', 'ない', 'いう', '言う','なん','いる','なる','こと','思う','お')
    
    if st.button('実行'):
      input_text = data[tg_col]
      input_text = ' '.join(input_text)
      doc = nlp(input_text)
      words = [token.lemma_ for token in doc
                if token.pos_ in include_pos and token.lemma_ not in stopwords]
      wc =WordCloud(background_color='white', font_path=font_path, regexp=r"[\w']+").generate(' '.join(words))
      plt.figure(figsize=(10, 5))
      plt.imshow(wc)
      plt.axis("off")
      

      # Output
      st.write(f""" #### word cloud""")
      st.pyplot(plt)
