import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import requests, json
import ast

class application_train:

    def __init__(self, train_path, ):
        self.path = train_path
        self.data = pd.read_csv(self.path, sep=',')

    def see_target(self):

        labels = list(map(str, self.data['TARGET'].unique().tolist()))
        values = self.data['TARGET'].value_counts().tolist()

        # pull is given as a fraction of the pie radius
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.1])])
        fig.update_layout(height = 400)
        return fig

    def missing_value(self, figsize = (6,8)):

        missing_count = self.data.isnull().sum()
        value_count = self.data.isnull().count()
        missing_percentage = round(missing_count / value_count * 100, 2)
        missing_df = pd.DataFrame(
            {'variable': list(self.data.columns), 'nbr_valeur_na': missing_count, 'percentage_na': missing_percentage})
        missing_df = missing_df.sort_values(by='nbr_valeur_na', ascending=False)
        missing_df = missing_df.reset_index(drop=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        col = list(missing_df.variable.tolist())
        nbr_data = list(len(self.data) - missing_df.nbr_valeur_na)
        ax.barh(col, nbr_data)
        plt.title("nomdre de donnée par variable")
        return (missing_df, fig)

    def info_client(self):

        infos_descrip = ['SK_ID_CURR',
                         "DAYS_BIRTH",
                         "CODE_GENDER",
                         "CNT_CHILDREN",
                         "NAME_FAMILY_STATUS",
                         "NAME_HOUSING_TYPE",
                         "NAME_CONTRACT_TYPE",
                         "NAME_INCOME_TYPE",
                         "OCCUPATION_TYPE",
                         "AMT_INCOME_TOTAL"
                         ]
        self.data_info = self.data[infos_descrip].set_index('SK_ID_CURR')

    def choice_variable(self):

        col_choice = ['SK_ID_CURR',
                      'TARGET',
                      'OWN_CAR_AGE',
                      'EXT_SOURCE_1',
                      'OCCUPATION_TYPE',
                      'DAYS_BIRTH',
                      'DAYS_EMPLOYED',
                      'REGION_RATING_CLIENT_W_CITY',
                      'REGION_RATING_CLIENT',
                      'NAME_INCOME_TYPE',
                      'EXT_SOURCE_3',
                      'EXT_SOURCE_2',
                      'NAME_EDUCATION_TYPE']
        self.data = self.data[col_choice]

    def handle_categorical_variable(self):
        col_cat = self.data.select_dtypes(include=["object"]).columns.tolist()
        le = LabelEncoder()
        for col in col_cat:
            if len(list(self.data[col].unique())) <= 2:
                le.fit(self.data[col])
                self.data[col] = le.transform(self.data[col])
        ohe = pd.get_dummies(self.data.select_dtypes(include=["object"]))
        self.data = pd.concat([self.data, ohe], axis=1)
        self.data.drop(col_cat, axis=1, inplace=True)

        col_choice_finale = [col for col in self.data.columns.tolist() if ' ' not in col]
        self.data = self.data[col_choice_finale]

    def handle_days_birth(self):
        self.data['DAYS_BIRTH'] = abs(self.data['DAYS_BIRTH']/365)

    def handle_days_employed(self):
        self.data['DAYS_EMPLOYED_ANOM'] = self.data["DAYS_EMPLOYED"] == 365243
        self.data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    def imputer_scaler(self, strategy):

        features = list(self.data.drop(columns=['TARGET', 'SK_ID_CURR']).columns)
        id = self.data['SK_ID_CURR']
        Target = self.data['TARGET']
        X = self.data[features]

        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataImputed = imputer.fit_transform(X)
        dataScaled = scaler.fit_transform(dataImputed)
        self.data = pd.DataFrame(dataScaled, columns=features)
        self.data['TARGET'] = Target
        self.data['SK_ID_CURR'] = id

    def prediction(self, id_client):
        ML_url ="https://fastapi-avi-oc-projet7.herokuapp.com/predict"
        data_to_predict = self.data[self.data["SK_ID_CURR"] == id_client].drop(['SK_ID_CURR', 'TARGET'], axis = 1)
        #data_to_predict = pd.DataFrame(self.data.drop(['SK_ID_CURR', 'TARGET'], axis = 1).iloc[id]).T
        data_to_predict_json = json.dumps(data_to_predict.to_dict('records')[0])
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", ML_url, headers=headers, data=data_to_predict_json)
        proba = float(ast.literal_eval(response.text)["proba_failed"])

        if proba > 0.7:
            message = "La probabilité de non remboursement est de: "
            result = "Ce client est à risque"
        else:
            message = "La probabilité de remboursement est de: "
            result = "Client éligible au prêt"
            proba = 1 - proba

        return (proba, result, message)

    def explain_model(self):
        shap_values = pickle.load(open(self.shap_path, 'rb'))
        return shap.summary_plot(shap_values, self.data)

    def explain_prediction(self, id_client):
        shap_values = pickle.load(open(self.shap_path, 'rb'))
        return (shap.force_plot(explainer.expected_value, shap_values[i], Xtest.iloc[i]))






