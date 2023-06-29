import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm

from matplotlib import pyplot as plt
import streamlit as st

st.set_page_config(layout='wide')

def load_data():
    df_raw = pd.read_csv('../data/df_ready.csv')
    return df_raw

def drop_columns(df_raw):
    df_raw = df_raw.drop(columns=['Unnamed: 0', 'Cluster',
        'condition','Disc_percentage', 'isSale', 'Imp_count', 'p_description',
       'currency', 'dateAdded', 'dateSeen', 'dateUpdated', 'imageURLs',
       'shipping', 'sourceURLs', 'weight', 'Date_imp_d.1',
       'Zscore_1','price_std'])

    df_raw.columns = ['date_imp','date_imp_d', 'category_name', 'name', 'price', 'disc_price', 'merchant', 'brand', 'manufacturer','day_n', 'month', 'month_n', 'day', 'week_number']
    return df_raw 

def change_dtypes(df1):
    df1['date_imp_d'] = pd.to_datetime(df1['date_imp_d'])
    return df1 


def data_preparation(df2):
    df_best = df2.loc[(df2['category_name'] == 'laptop, computer') & (df2['merchant'] == 'Bestbuy.com')]
    df_agg = df_best.groupby(['name','week_number']).agg({'disc_price':'mean', 'date_imp': 'count'}).reset_index()

    # orientando pelo preço
    x_price = df_agg.pivot(index='week_number',columns='name',values='disc_price')

    #orientando pela demanda
    y_demand = df_agg.pivot(index='week_number',columns='name',values='date_imp')

    median_price = np.round(x_price.median(),2)
    x_price.fillna(median_price, inplace=True)
    y_demand.fillna(0, inplace=True)
    return x_price, y_demand 

def ml_elasticity(x_price, y_demand):
    results_values_laptop = {
    "name": [],
    "price_elastity": [],
    "price_mean": [],
    "quantity_mean": [],
    "intercept": [],
    "slope": [],
    "rsquared": [],
    "p_value": []

    }

    df = pd.DataFrame()
    for column in x_price.columns:
        #print(column)
        list_price = []
        list_demand = []
        column_points = []
        for i in range(len(x_price[column])):
            column_points.append({'x_price': x_price.reset_index(drop=True)[column][i],
                                'y_demand': y_demand.reset_index(drop=True)[column][i]})
            
            list_price.append(column_points[i]['x_price'])
            list_demand.append(column_points[i]['y_demand'])


        X = sm.add_constant(list_price)
        model = sm.OLS(list_demand, X)
        results = model.fit()

        if results.f_pvalue < 0.05:
            rsquared = results.rsquared
            pvalue = results.f_pvalue
            intercept, slope = results.params
            
            mean_price = np.round(np.mean(list_price),2)
            mean_quantity = np.round(np.mean(list_demand),2)


            price_elastity = slope*(mean_price/mean_quantity)

            results_values_laptop['name'].append(column)
            results_values_laptop['price_elastity'].append(price_elastity)
            results_values_laptop['rsquared'].append(rsquared)
            results_values_laptop['p_value'].append(pvalue)
            results_values_laptop['intercept'].append(intercept)
            results_values_laptop['slope'].append(slope)
            results_values_laptop['price_mean'].append(mean_price)
            results_values_laptop['quantity_mean'].append(mean_quantity)
        df_elastity = pd.DataFrame(results_values_laptop)
    return df_elastity

def simulation_elasticity(percentual, x_price, y_demand, df_elastity, op):
    if percentual != 0:
        result_faturamento = {
        'name': [],
        'faturamento_atual': [],
        'faturamento_alteracao': [],
        'alteracao_faturamento': [],
        'faturamento_novo': [],
        'variacao_faturamento': [],
        'variacao_percentual': []
        }

        if op == 'Desconto':
            percentual = -percentual
        

        for i in range(len(df_elastity)):

            #print(i)
            preco_atual_medio = df_elastity['price_mean'][i]
            demanda_atual = y_demand[df_elastity['name'][i]].sum()

            if percentual < 0:
                alteracao_preco = preco_atual_medio*(1-((percentual*(-1))/100))
                
            if percentual > 0:
                alteracao_preco = (preco_atual_medio*percentual) + preco_atual_medio
            
            aumento_demanda = (percentual/100)*df_elastity['price_elastity'][i]

            demanda_nova = aumento_demanda*demanda_atual

            faturamento_atual = round(preco_atual_medio*demanda_atual, 2)
            faturamento_novo = round(alteracao_preco*demanda_nova, 2)

            if percentual < 0:
                faturamento_alteracao = round(faturamento_atual*(1-((percentual*(-1))/100)), 2)
                #print('FATURAMENTO REDUCAO:{}', faturamento_alteracao)
                alteracao_faturamento = round(faturamento_atual-faturamento_alteracao, 2)
            if percentual > 0:
                faturamento_alteracao = (faturamento_atual*percentual) + faturamento_atual
                alteracao_faturamento = round(faturamento_atual+faturamento_alteracao, 2)

            variacao_faturamento = round(faturamento_novo-faturamento_atual ,2)

            variacao_percentual = round(((faturamento_novo-faturamento_atual)/faturamento_atual),2)


            result_faturamento['name'].append(df_elastity['name'][i])
            result_faturamento['faturamento_atual'].append(faturamento_atual)
            result_faturamento['faturamento_alteracao'].append(faturamento_alteracao)
            result_faturamento['alteracao_faturamento'].append(alteracao_faturamento)
            result_faturamento['faturamento_novo'].append(faturamento_novo)
            result_faturamento['variacao_faturamento'].append(variacao_faturamento)
            result_faturamento['variacao_percentual'].append(variacao_percentual)

        resultado = pd.DataFrame(result_faturamento)
        return resultado
    else:
        return None


if __name__ == '__main__':
    df_raw = load_data()
    df1 = drop_columns(df_raw)

    df2 = change_dtypes(df1)

    x_price, y_demand = data_preparation(df2)

    df_elasticity = ml_elasticity(x_price, y_demand)

    tab1, tab2 = st.tabs(['Elasticidade de Preço', 'Simule Cenários'])


    with tab1:
        st.header('Elasticidade de Preço dos Produtos')
        st.dataframe(df_elasticity)
    with tab2:

        col1, col2 = st.columns((1,1))
        with col1:
            #st.markdown("## Você gostaria de aplicar um desconto ou um aumento de preço nos produtos?")
            st.markdown("<h2 style='text-align: center;'>Você gostaria de aplicar um desconto ou um aumento de preço nos produtos?</h2>", unsafe_allow_html=True)
            option = st.selectbox(
            '',
            ('Aumento de Preço', 'Aplicar Desconto'))
            if option == 'Aumento de Preço':
                op = 'Aumento de Preço'
            else:
                op = 'Desconto'

        with col2:
            #st.markdown('## Qual o percentual de '+op+' que você deseja aplicar?')
            st.markdown('<h2 style="text-align: center;">Qual o percentual de '+op+' que você deseja aplicar?</h2>', unsafe_allow_html=True)
            number = st.number_input('')


        if number != 0:

            final = simulation_elasticity(number, x_price, y_demand, df_elasticity, op)
            st.dataframe(final)
