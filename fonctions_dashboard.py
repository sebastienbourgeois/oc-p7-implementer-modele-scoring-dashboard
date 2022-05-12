import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@st.cache
def charger_demandes_credit(chemin_fichier_donnees):
	variables_a_conserver = [
		'SK_ID_CURR',
		'NAME_CONTRACT_TYPE',
		'CODE_GENDER',
		'FLAG_OWN_CAR',
		'FLAG_OWN_REALTY',
		'AMT_INCOME_TOTAL',
		'AMT_CREDIT',
		'NAME_INCOME_TYPE',
		'NAME_EDUCATION_TYPE',
		'NAME_FAMILY_STATUS',
		'NAME_HOUSING_TYPE',
		'CNT_FAM_MEMBERS',
		'DEF_30_CNT_SOCIAL_CIRCLE',
		'DAYS_BIRTH',
		'OWN_CAR_AGE',
		'DAYS_EMPLOYED',
		'AMT_ANNUITY'
	]
	return pd.read_csv(chemin_fichier_donnees, usecols=variables_a_conserver)

@st.cache
def generer_features_engineering(df_demandes_credit_brutes):
	df_demandes_credit = df_demandes_credit_brutes.copy()
	df_demandes_credit = calculer_age_client(df_demandes_credit)
	df_demandes_credit = calculer_duree_emploi(df_demandes_credit)
	df_demandes_credit = definir_anciennete_voiture(df_demandes_credit)
	df_demandes_credit = definir_anciennete_emploi(df_demandes_credit)
	df_demandes_credit = calculer_taux_remboursement_annuel(df_demandes_credit)
	return df_demandes_credit

def calculer_age_client(df_demandes_credit):
	df_demandes_credit["CLIENT_AGE"] = round(-df_demandes_credit['DAYS_BIRTH']/365, 0)
	df_demandes_credit.drop(columns=["DAYS_BIRTH"], inplace=True)
	return df_demandes_credit

def calculer_duree_emploi(df_demandes_credit):
	df_demandes_credit["EMPLOYMENT_DURATION"] = round(-df_demandes_credit['DAYS_EMPLOYED']/365, 0)
	df_demandes_credit.drop(columns=["DAYS_EMPLOYED"], inplace=True)
	return df_demandes_credit

def definir_anciennete_voiture(df_demandes_credit):
	masque_pas_voiture = df_demandes_credit['OWN_CAR_AGE'].isnull()
	masque_voiture_neuve = df_demandes_credit['OWN_CAR_AGE'] <= 3
	masque_jeune_voiture = (df_demandes_credit['OWN_CAR_AGE'] >= 4) & (df_demandes_credit['OWN_CAR_AGE'] <= 9)
	masque_vieille_voiture = (df_demandes_credit['OWN_CAR_AGE'] >= 10) & (df_demandes_credit['OWN_CAR_AGE'] <= 19)
	masque_tres_vieille_voiture = df_demandes_credit['OWN_CAR_AGE'] >= 20

	df_demandes_credit.loc[masque_pas_voiture, 'OWN_CAR_TYPE'] = "No car"
	df_demandes_credit.loc[masque_voiture_neuve, 'OWN_CAR_TYPE'] = "New car"
	df_demandes_credit.loc[masque_jeune_voiture, 'OWN_CAR_TYPE'] = "Young car"
	df_demandes_credit.loc[masque_vieille_voiture, 'OWN_CAR_TYPE'] = "Old car"
	df_demandes_credit.loc[masque_tres_vieille_voiture, 'OWN_CAR_TYPE'] = "Very old car"

	df_demandes_credit.drop(columns=["OWN_CAR_AGE"], inplace=True)

	return df_demandes_credit

def definir_anciennete_emploi(df_demandes_credit):
	masque_sans_activite = df_demandes_credit['EMPLOYMENT_DURATION'] == -1001
	masque_debutants = (df_demandes_credit['CLIENT_AGE'] <= 29) & (df_demandes_credit['EMPLOYMENT_DURATION'] >= 0) & (df_demandes_credit['EMPLOYMENT_DURATION'] <= 3)
	masque_nouveau_job = (df_demandes_credit['CLIENT_AGE'] >= 30) & (df_demandes_credit['EMPLOYMENT_DURATION'] >= 0) & (df_demandes_credit['EMPLOYMENT_DURATION'] <= 3)
	masque_confirmes = (df_demandes_credit['EMPLOYMENT_DURATION'] > 3) & (df_demandes_credit['EMPLOYMENT_DURATION'] <= 10)
	masque_anciens = df_demandes_credit['EMPLOYMENT_DURATION'] > 10

	df_demandes_credit.loc[masque_sans_activite, 'JOB_SENIORITY'] = "No job"
	df_demandes_credit.loc[masque_debutants, 'JOB_SENIORITY'] = "Beginner"
	df_demandes_credit.loc[masque_nouveau_job, 'JOB_SENIORITY'] = "New job"
	df_demandes_credit.loc[masque_confirmes, 'JOB_SENIORITY'] = "Medium seniority"
	df_demandes_credit.loc[masque_anciens, 'JOB_SENIORITY'] = "Long seniority"

	df_demandes_credit.drop(columns=["EMPLOYMENT_DURATION"], inplace=True)

	return df_demandes_credit

def calculer_taux_remboursement_annuel(df_demandes_credit):
	df_demandes_credit['ANNUAL_PAYMENT_RATE'] = df_demandes_credit['AMT_ANNUITY']/df_demandes_credit['AMT_CREDIT']
	df_demandes_credit.drop(columns=["AMT_ANNUITY"], inplace=True)
	return df_demandes_credit

@st.cache
def standardiser_data(df_demandes_credit):
	df_demandes_credit_sans_ID = df_demandes_credit.drop(columns=['SK_ID_CURR'])
	pipeline_pretraitements = creer_pipeline_pretraitements(df_demandes_credit_sans_ID)
	nom_colonnes = renommer_colonnes(df_demandes_credit_sans_ID, pipeline_pretraitements)
	df_std_demandes_credit = pd.DataFrame(data=pipeline_pretraitements.transform(df_demandes_credit_sans_ID), 
										  columns=nom_colonnes)
	return df_std_demandes_credit

def creer_pipeline_pretraitements(df_demandes_credit):
	s = (df_demandes_credit.dtypes == 'object')
	variables_categorielles = list(s[s].index)

	categorical_transformer = ColumnTransformer(
	    transformers=[
	        ('categorielles', OneHotEncoder(handle_unknown='ignore', sparse=False), variables_categorielles)
	    ],
	    remainder = 'passthrough'
	)

	preprocessor = Pipeline(steps=[
	    ('encodage', categorical_transformer),
	    ('standardisation', StandardScaler(with_mean=False))
	])

	return preprocessor.fit(df_demandes_credit)

def renommer_colonnes(df_demandes_credit, pipeline_pretraitements):
	nom_colonnes_pipeline = pipeline_pretraitements.get_feature_names_out(df_demandes_credit.columns)
	nom_colonnes = []
	for colonne in nom_colonnes_pipeline:
	    if colonne[0:13] == "categorielles":
	        nom_colonnes.append(colonne[15:])
	    else:
	        nom_colonnes.append(colonne[11:])
	return nom_colonnes

def ajouter_donnees_manquantes(df_std_demandes_credit, df_demandes_credit):
	df_std_demandes_credit = ajouter_id_client(df_std_demandes_credit, df_demandes_credit)
	df_std_demandes_credit['NAME_INCOME_TYPE_Maternity leave'] = 0
	return df_std_demandes_credit

def ajouter_id_client(df_std_demandes_credit, df_demandes_credit):
	df_id_client = pd.DataFrame(df_demandes_credit['SK_ID_CURR'])
	return df_id_client.merge(df_std_demandes_credit, how='inner', left_index=True, right_index=True)

def recuperer_liste_id_clients(df_demandes_credit):
	return df_demandes_credit['SK_ID_CURR'].tolist()

def recuperer_liste_variables(df_demandes_credit):
	liste_variables = df_demandes_credit.columns.tolist()
	liste_variables.remove('SK_ID_CURR')
	return liste_variables

def recuperer_donnees_std_client(id_client, df_std_demandes_credit):
	df_tmp = df_std_demandes_credit[df_std_demandes_credit['SK_ID_CURR'] == id_client]
	return df_tmp.drop(columns=['SK_ID_CURR']).values.tolist()

def recuperer_prediction_client(donnees_client):
	headers = {"Content-Type": "application/json"}
	#URI = 'http://127.0.0.1:5000/predictions'
	URI = 'https://oc-api-modele-scoring.herokuapp.com/predictions'

	data_client_json = {'std_donnees_client': donnees_client}
	response = requests.post(headers=headers, url=URI, json=data_client_json)

	if response.status_code != 200:
			raise Exception(
					"Request failed with status {}, {}".format(response.status_code, response.text))

	return response.json()

def construire_jauge_score(score_remboursement_client):
	fig = go.Figure(
		go.Indicator(
		    mode = "gauge+number",
		    value = score_remboursement_client,
		    domain = {'x': [0, 1], 'y': [0, 1]},
		    title = {'text': "Probabilité de remboursement du crédit"},
		    gauge = {
		    	'axis': {'range': [None, 1]},
		    	'steps': [
		    		{'range': [0, 0.35], 'color': "tomato"},
		    		{'range': [0.65, 1], 'color': "lightgreen"}
		    	]
		    }
	    )
	)
	return fig

def construire_graphique(df_demandes_credit, variable):
	fig = go.Figure(data=[go.Histogram(x=df_demandes_credit[variable])])
	return fig

def recuperer_donnee_client(df_demandes_credit, id_client, variable):
	df_tmp = df_demandes_credit[df_demandes_credit['SK_ID_CURR'] == id_client]
	df_tmp = df_tmp.reset_index(drop=True)
	donnee_client = df_tmp.loc[0, variable]
	return donnee_client

def ajouter_position_client(graphique, valeur_client):
	graphique.add_annotation(x=valeur_client, y=0, text="<b>Position du client</b>", 
		showarrow=True, arrowhead=1, arrowwidth=2,
		bordercolor="black", borderwidth=1,
		bgcolor="white")
	return graphique