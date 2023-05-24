import pandas as pd
from pandas.core.frame import DataFrame
import streamlit as st
import openpyxl

from mlxtend.frequent_patterns import apriori #頻出アイテム集合を抽出する関数
from mlxtend.frequent_patterns import association_rules

st.set_page_config(page_title='アソシエーション分析')
st.markdown('## アソシエーション分析/品番３ケタ')

@st.cache_data
def make_data(file):
    df = pd.read_excel(
    file, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 3, 9, 10, 15, 42]) #index　ナンバー不要　index_col=0

    df['品番2'] = df['商品コード'].map(lambda x:str(x)[0:3])
    df['伝票番号2'] = df['伝票番号'].map(lambda x:str(x)[0:8])

    temp1 = df.groupby(["伝票番号2", "品番2"])["数量"].sum()

    #行から列へピボット: unstack()
    temp2 = temp1.unstack().fillna(0) 

    association_df = temp2.apply(lambda x: x>0) 

    freq_items1 = apriori(association_df, min_support=0.001, use_colnames=True) 
    # min_support 閾値 その組み合わせの全体の構成比
    freq_items1.sort_values('support', ascending=False) 

    st.write(freq_items1)

    #support 「AとBが一緒にあるデータの数」/「全てのデータ数」
    #confidence 商品Aが買われた中で、商品Bも一緒に買われた割合
    #lift 「確信度」/「Bの起こる確率」

    rules = association_rules(freq_items1, metric='lift', min_threshold=1)
    #liftが1より大きい組み合わせを抽出
    rules = rules.sort_values('lift', ascending=False)
    
    rules2 = rules[(rules['antecedent support'] >= 0.005) & (rules['consequent support'] >= 0.005)]
    rules2 = rules2[(rules2['confidence'] >= 0.3)]
    st.write(rules2)
    st.write(len(rules2))
    #antecedents 条件部　consequents 結論部



# ***ファイルアップロード 今期***
uploaded_file = st.sidebar.file_uploader('今期', type='xlsx', key='now')
df = DataFrame()
if uploaded_file:
    df_now = make_data(uploaded_file)

else:
    st.info('今期のファイルを選択してください。')