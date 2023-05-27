import pandas as pd
from pandas.core.frame import DataFrame
import streamlit as st
# import openpyxl

from mlxtend.frequent_patterns import apriori #頻出アイテム集合を抽出する関数
from mlxtend.frequent_patterns import association_rules

st.set_page_config(page_title='アソシエーション分析')
st.markdown('## アソシエーション分析')

@st.cache_data
def make_data(file):
    df = pd.read_excel(
    file, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 3, 9, 10, 15, 42]) #index　ナンバー不要　index_col=0
    
    return df


# ***ファイルアップロード 今期***
uploaded_file = st.sidebar.file_uploader('今期', type='xlsx', key='full')
df_rules = DataFrame()
if uploaded_file:
    df_rules = make_data(uploaded_file)

else:
    st.info('今期のファイルを選択してください。')
    st.stop()

#************************項目選択
selected_data = st.selectbox(
    'データ選択',
    ['--', 'フル品番: 例SG261A', '頭品番＋１ケタ: 例SG2'],
    key='sd')

if selected_data == '--':
    st.info('データを選択してください')
    st.stop()


elif selected_data == 'フル品番: 例SG261A':

    df_rules['品番2'] = df_rules['商品コード'].map(lambda x:str(x).split(' ')[0])
    df_rules['伝票番号2'] = df_rules['伝票番号'].map(lambda x:str(x)[0:8])

    temp1 = df_rules.groupby(["伝票番号2", "品番2"])["数量"].sum()

    #行から列へピボット: unstack()
    temp2 = temp1.unstack().fillna(0) 

    association_df = temp2.apply(lambda x: x>0) 

    freq_items1 = apriori(association_df, min_support=0.001, use_colnames=True) 
    # min_support 閾値 その組み合わせの全体の構成比
    freq_items1.sort_values('support', ascending=False) 

    #■ support 「AとBが一緒にあるデータの数」/「全てのデータ数」
    #■ confidence 商品Aが買われた中で、商品Bも一緒に買われた割合
    #■ lift 「確信度」/「Bの起こる確率」

    df_rules = association_rules(freq_items1, metric='lift', min_threshold=1)
    #liftが1より大きい組み合わせを抽出
    df_rules = df_rules.sort_values('lift', ascending=False)

    with st.expander('項目の説明', expanded=False):
        st.caption('support 「AとBが一緒にあるデータの数」/「全てのデータ数」')
        st.caption('confidence 商品Aが買われた中で、商品Bも一緒に買われた割合')
        st.caption('lift 「確信度」/「Bの起こる確率」')

    selected_cate = st.selectbox(
        '項目選択',
        ['--', '分析一覧を見る', '品番で抽出']
        ,key='sc'
    )

    if selected_cate == '--':
        st.info('項目を選択してください')
        st.stop()

    elif selected_cate == '分析一覧を見る':
        with st.form("入力欄"):
            lower_antecedent = st.number_input('antecedent support下限')
            lower_consequent = st.number_input('consequent support下限')
            lower_lift = st.number_input('lift下限', value=1.5)

            submitted = st.form_submit_button("Submit")

            if submitted:
                df_rules2 = df_rules[(df_rules['antecedent support'] >= lower_antecedent) & \
                                    (df_rules['consequent support'] >= lower_consequent)]
                df_rules2 = df_rules2[(df_rules2['lift'] >= lower_lift)]

                
                st.dataframe(df_rules2)
                st.write(f'データ数: {len(df_rules2)}')

    elif selected_cate == '品番で抽出':
        selected_item = st.text_input('品番を入力')

        if selected_item == '':
            st.info('品番を入力してください ※半角英数/アルファベットは大文字　例SG261A')
            st.stop()

        else:
            df_rules2 = df_rules[df_rules['antecedents'] ==frozenset({selected_item})]
            st.dataframe(df_rules2)

#**********************************************************************************************************
elif selected_data == '頭品番＋１ケタ: 例SG2':

        df_rules['品番2'] = df_rules['商品コード'].map(lambda x:str(x)[0:3])
        df_rules['伝票番号2'] = df_rules['伝票番号'].map(lambda x:str(x)[0:8])

        temp1 = df_rules.groupby(["伝票番号2", "品番2"])["数量"].sum()

        #行から列へピボット: unstack()
        temp2 = temp1.unstack().fillna(0) 

        association_df = temp2.apply(lambda x: x>0) 

        freq_items1 = apriori(association_df, min_support=0.001, use_colnames=True) 
        # min_support 閾値 その組み合わせの全体の構成比
        freq_items1.sort_values('support', ascending=False) 

        #■ support 「AとBが一緒にあるデータの数」/「全てのデータ数」
        #■ confidence 商品Aが買われた中で、商品Bも一緒に買われた割合
        #■ lift 「確信度」/「Bの起こる確率」

        df_rules = association_rules(freq_items1, metric='lift', min_threshold=1)
        #liftが1より大きい組み合わせを抽出
        df_rules = df_rules.sort_values('lift', ascending=False)

        with st.expander('項目の説明', expanded=False):
            st.caption('support 「AとBが一緒にあるデータの数」/「全てのデータ数」')
            st.caption('confidence 商品Aが買われた中で、商品Bも一緒に買われた割合')
            st.caption('lift 「確信度」/「Bの起こる確率」')

        selected_cate = st.selectbox(
            '項目選択',
            ['--', '分析一覧を見る', '品番で抽出']
            ,key='sc2'
        )

        if selected_cate == '--':
            st.info('項目を選択してください')
            st.stop()

        elif selected_cate == '分析一覧を見る':
            with st.form("入力欄"):
                lower_antecedent = st.number_input('antecedent support下限')
                lower_consequent = st.number_input('consequent support下限')
                lower_lift = st.number_input('lift下限', value=1.5)

                submitted = st.form_submit_button("Submit")

                if submitted:
                    df_rules2 = df_rules[(df_rules['antecedent support'] >= lower_antecedent) & \
                                        (df_rules['consequent support'] >= lower_consequent)]
                    df_rules2 = df_rules2[(df_rules2['lift'] >= lower_lift)]

                    
                    st.dataframe(df_rules2)
                    st.write(f'データ数: {len(df_rules2)}')

        elif selected_cate == '品番で抽出':
            selected_item = st.text_input('品番を入力')

            if selected_item == '':
                st.info('品番を入力してください ※半角英数/アルファベットは大文字 例SG2')
                st.stop()

            else:
                df_rules2 = df_rules[df_rules['antecedents'] ==frozenset({selected_item})]
                st.dataframe(df_rules2)

