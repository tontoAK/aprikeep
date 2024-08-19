import streamlit as st
from PIL import Image

st.title("口コミ評判アプリ")
st.write("""### サイドバーより、ポジネガ分析を選択して下さい""")
image = Image.open('AI6.jpg')
st.image(image, caption='自然言語処理',width=300)