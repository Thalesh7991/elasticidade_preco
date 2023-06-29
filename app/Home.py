import streamlit as st

st.set_page_config(
    page_title="Previsão de Elasticidade de Preço",
    page_icon="💰",
)

st.title("Bem-vindo(a) à Aplicação de Previsão de Elasticidade de Preço! 💰")

st.markdown(
    """
    Esta aplicação foi desenvolvida para um e-commerce de produtos eletrônicos que solicitou um modelo de Inteligência Artificial capaz de prever a elasticidade de preço. 
    Com esse modelo, você pode simular cenários de aumento de preço e descontos e receber feedbacks em tempo real sobre os impactos financeiros no faturamento.
    
    ### Como funciona?
    - Carregue os dados dos produtos eletrônicos do e-commerce
    - Treine o modelo de IA para prever a elasticidade de preço desses produtos
    - Simule cenários de aumento de preço e descontos
    - Analise o impacto financeiro na receita do e-commerce
    
    **👈 Selecione uma demonstração na barra lateral** para ver exemplos e experimentar você mesmo!
    """
)

