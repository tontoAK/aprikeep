#初期設定
import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt

st.title("word count")

# Input
nlp = spacy.load('ja_ginza')
pos_dic = {'名詞':'NOUN', '代名詞':'PRON', 
           '固有名詞':'PROPN','動詞':'VERB','形容詞':'ADJ'}
stopwords=['お','個','円','ない','方','つ','ある','いる','なる','以上']

#ファイル読込
uploaded_file = st.file_uploader("Excelを選択", type='xlsx')
#品詞設定
select_pos = st.sidebar.multiselect('品詞選択',
                ['名詞','代名詞','固有名詞','動詞','形容詞'],
                ['名詞'])

# Process
if uploaded_file is not None:
  #data = pd.read_csv(uploaded_file)
  data= pd.read_excel(uploaded_file)
  tg_col = st.selectbox('対象列選択,テキスト以外または容量制限でエラーになります',data.columns)

  if st.button('実行'):
    if tg_col is not None:
      data = data.dropna()
      input_text = data[tg_col]
      input_text = ' '.join(input_text)
      doc = nlp(input_text)
      output_word = []
      tg_pos = [pos_dic[x] for x in select_pos]
      for token in doc:
        if token.pos_ in tg_pos:
          if token.lemma_ not in stopwords:
            output_word.append(token.lemma_)
      output_df = pd.DataFrame({'Word':output_word})
      output_df = output_df.groupby('Word',as_index=False).size()
      output_df.set_index('Word', inplace=True)
      output_df.sort_values('size',ascending=False,inplace=True)

      # Output
      st.write(f""" #### word Analysis (原文）""")
      st.dataframe(data)
      st.write(f" word {select_pos} count 30")
      st.dataframe(output_df[:30])
      st.write(f" word {select_pos} 20 count")
      st.bar_chart(output_df[:20])
      #st.pyplot(plt)

      


