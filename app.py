import streamlit as st
from application_train import application_train


def app():

    st.set_page_config(
        page_title="Prêt à dépenser - Plateforme d'octroi de crédit",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.image("pretadep.png")
    st.sidebar.title("Filtre")
    st.write("# Scoring App: Approbation d'un prêt")


    train = application_train("application_train.csv")
    train.info_client()
    col1, col2 = st.columns(2)
    with col1:
        st.write("Aperçu des données d'entrée")
        st.dataframe(train.data.head(20))
    with col2:
        st.write("Proportion des remboursements des prêts")
        st.plotly_chart(train.see_target())

    expander_na = st.expander("voir les informations sur les valeurs manquantes")
    expander_na.markdown("<h1 style='text-align: center; color: red;'>Valeurs Manquantes</h1>", unsafe_allow_html=True)
    expander_na.dataframe(train.missing_value()[0])


    train.choice_variable()
    train.handle_categorical_variable()
    train.handle_days_birth()
    train.handle_days_employed()
    train.imputer_scaler('median')


    id_client = st.sidebar.selectbox('Select ID Client :', train.data["SK_ID_CURR"].tolist())
    st.sidebar.subheader(f'Les informations du client {id_client} : ')
    st.sidebar.table(train.data_info.astype(str).loc[id_client][1:9])


    proba, result, message= train.prediction(id_client)
    expander_pred = st.expander("voir la prévision de solvabilité")

    if result == "Ce client est à risque":
        expander_pred.write(message + str(round(proba,3)*100) + "%")
        expander_pred.error(result)
    else:
        expander_pred.write(message + str(round(proba,3)*100) + "%")
        expander_pred.success(result)

    expander_explain = st.expander("voir la contribution des variables sur la prévision")
    expander_explain.image("SHAP.JPG")

    expander_client = st.expander("voir les informations concernant les clients similaires")
    expander_client.write(f'### Les autres clients similaires {id_client}')

    feature_name = expander_client.selectbox('Selecting feature name :',
                                ["CODE_GENDER",
                                 "CNT_CHILDREN",
                                 "NAME_FAMILY_STATUS",
                                 "NAME_HOUSING_TYPE",
                                 "NAME_CONTRACT_TYPE",
                                 "NAME_INCOME_TYPE",
                                 "OCCUPATION_TYPE"])
    data_feature = train.data_info[feature_name].value_counts(normalize=True)
    expander_client.bar_chart(data_feature)





app()