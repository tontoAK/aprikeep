#初期設定
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import networkx as nx
from pyvis.network import Network
import japanize_matplotlib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from PIL import Image


#警告を出ないようにする
import warnings
warnings.filterwarnings('ignore')

st.title("共起ネットワーク分析")

# input
# サイドバーの入力
size = st.sidebar.number_input("データ表示調整数",0.025,0.080,0.025)
#データ読込
uploaded_file = st.file_uploader("Excelを選択", type='xlsx')
if uploaded_file is not None:
  #データ読込　列のselectBox作成
  df = pd.read_excel(uploaded_file)
  tg_col = st.selectbox('対象列選択,テキスト以外はエラーになります',df.columns)
 
  # 使用する単語の品詞とストップワードの指定
  nlp = spacy.load('ja_ginza')
  include_pos =('NOUN', 'VERB', 'ADJ')+('PROPN', 'ADV')
  stopwords = ('する', 'ある', 'ない', '方', 'お','もの', '個','円','なる','こと','思う','もう','いる','しまう')
  #process
  if st.button('実行'):
    def extract_words(sent, pos_tags, stopwords):
        words = [token.lemma_ for token in sent
             if token.pos_ in pos_tags and token.lemma_ not in stopwords]
        return words
    def count_cooccurrence(sents, token_length='{2,}'):
        token_pattern=f'\\b\\w{token_length}\\b'
        count_model = CountVectorizer(token_pattern=token_pattern)
        X = count_model.fit_transform(sents)
        words = count_model.get_feature_names_out()
        word_counts = np.asarray(X.sum(axis=0)).reshape(-1)
        X[X > 0] = 1
        Xc = (X.T * X)
        return words, word_counts, Xc, X

    def word_weights(words, word_counts):
        count_max = word_counts.max()
        weights = [(word, {'weight': count / count_max})
                   for word, count in zip(words, word_counts)]
        return weights
    
    def cooccurrence_weights(words, Xc, weight_cutoff):
        Xc_max = Xc.max()
        cutoff = weight_cutoff * Xc_max
        weights = [(words[i], words[j], Xc[i,j] / Xc_max)
                    for i, j in zip(*Xc.nonzero()) if i < j and Xc[i,j] > cutoff]
        return weights
    def create_network(words, word_counts, Xc, weight_cutoff):
        G = nx.Graph()
        weights_w = word_weights(words, word_counts)
        G.add_nodes_from(weights_w)
        weights_c = cooccurrence_weights(words, Xc, weight_cutoff)
        G.add_weighted_edges_from(weights_c)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G
    def pyplot_network(G):
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=0.1)
        weights_n = np.array(list(nx.get_node_attributes(G, 'weight').values()))
        nx.draw_networkx_nodes(G, pos, node_size=300 * weights_n)
        weights_e = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
        nx.draw_networkx_edges(G, pos, width=20 * weights_e)
        nx.draw_networkx_labels(G, pos, font_family='IPAexGothic')
        plt.axis("off")
        plt.show()
    def nx2pyvis_G(G):
        pyvis_G = Network(width='800px', height='800px',bgcolor="#FFFFFF", notebook=True,cdn_resources='in_line')#remote in_line
        for node, attrs in G.nodes(data=True):
            pyvis_G.add_node(node, title=node, size=30 * attrs['weight'])
        for node1, node2, attrs in G.edges(data=True):
            pyvis_G.add_edge(node1, node2, width=20 * attrs['weight'])
        return pyvis_G
    # 口コミを解析し共起を算出
    sents = []
    for doc in nlp.pipe(df['comment']):
        sents.extend([' '.join(extract_words(sent, include_pos, stopwords))
                      for sent in doc.sents])
    words, word_counts, Xc, X = count_cooccurrence(sents,'{1,}')

    # ネットワークの生成
    G = create_network(words, word_counts, Xc, size)
    # 静的ネットワークの描画
    pyplot_network(G)
    # 動的ネットワークの描画
    #pyvis_G = nx2pyvis_G(G)
    #pyvis_G.show_buttons()
    #pyvis_G.show('mygraph.html')

    #output
    st.write(f""" ### 原文""")
    st.dataframe(df)
    #グラフ表示
    st.write(f""" ### ネットワーク図""")
    st.pyplot(plt)
    #st.markdown("""pyvis_G""",unsafe_allow_html=True)
