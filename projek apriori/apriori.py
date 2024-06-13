import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# load dataset
df = pd.read_csv("D:/BREAD-APRIORI/bread basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

# Add period of day and weekday/weekend columns for filtering
df['period_day'] = df['date_time'].apply(lambda x: 'Morning' if 5 <= x.hour < 12 else 'Afternoon' if 12 <= x.hour < 17 else 'Evening' if 17 <= x.hour < 21 else 'Night')
df['weekday_weekend'] = df['date_time'].apply(lambda x: 'Weekend' if x.weekday() >= 5 else 'Weekday')

df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday

df['month'] = df['month'].replace([i for i in range(1, 13)], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
df['day'] = df['day'].replace([i for i in range(7)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

st.title('Market Basket Analysis With Apriori')

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data['period_day'].str.contains(period_day, case=False, na=False)) & 
        (data['weekday_weekend'].str.contains(weekday_weekend, case=False, na=False)) &
        (data['month'].str.contains(month, case=False, na=False)) &
        (data['day'].str.contains(day, case=False, na=False))
    ]
    return filtered if filtered.shape[0] > 0 else 'No Result!'

def user_input_features():
    item = st.selectbox('Item', sorted(df['Item'].unique()))
    period_day = st.selectbox('Period Day', ['Morning','Afternoon','Evening','Night'])
    weekday_weekend = st.selectbox('Weekday / Weekend', ['Weekend','Weekday'])
    month = st.select_slider('Month',["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day',["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value='Sat')
    return period_day, weekday_weekend, month, day, item

period_day, weekday_weekend, month, day, item = user_input_features()

data = get_data(period_day, weekday_weekend, month, day)

def encode(x):
    return 1 if x >= 1 else 0

if type(data) != str:
    item_count = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)
    
    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
    
    metric = 'lift'
    min_threshold = 1
    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[['antecedents','consequents','support','confidence','lift']]
    rules.sort_values('confidence', ascending=False, inplace=True)

    def parse_list(x):
        return ', '.join(list(x))

    def return_item_df(item_antecedents):
        result = rules[rules['antecedents'].apply(lambda x: item_antecedents in list(x))]
        if not result.empty:
            antecedents = parse_list(result.iloc[0]['antecedents'])
            consequents = parse_list(result.iloc[0]['consequents'])
            return antecedents, consequents
        return None, None

    antecedents, consequents = return_item_df(item)

    if antecedents:
        st.markdown('Hasil Rekomendasi : ')
        st.success(f'Jika konsumen membeli **{item}** , maka membeli **{consequents}** secara bersamaan')
    else:
        st.markdown('Tidak ada rekomendasi untuk item ini.')

else:
    st.markdown('No Result!')
