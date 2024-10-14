#初期設定
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
st.title("word cloud")

# Input
uploaded_file = st.file_uploader("CSVを選択", type='csv')
#uploaded_file = st.file_uploader("Excelを選択", type='xlsx')

# Process
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
  #data= pd.read_excel(uploaded_file)
  tg_col = st.selectbox('対象列選択,テキスト以外または容量制限でエラーになります',data.columns)
    
  if st.button('実行'):
    input_text = data[tg_col]
      
    tw_words = ' '.join(input_text)
    tw_wc = WordCloud()
    tw_wc.generate(tw_words)
    plt.figure(figsize=(10, 5))
    plt.imshow(tw_wc)
    plt.axis("off")
      

    # Output
    st.write(f""" #### word cloud""")
    st.pyplot(plt)
