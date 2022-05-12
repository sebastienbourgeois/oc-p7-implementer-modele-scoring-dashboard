import streamlit as st
import fonctions_dashboard as fd

def main():
	# Chargement et préparation des données
	df_demandes_credit_brutes = fd.charger_demandes_credit("data/application_test.csv")
	df_demandes_credit = fd.generer_features_engineering(df_demandes_credit_brutes)
	df_std_demandes_credit = fd.standardiser_data(df_demandes_credit)
	df_std_demandes_credit = fd.ajouter_donnees_manquantes(df_std_demandes_credit, df_demandes_credit)

	# Sélection par l'utilisateur des données client à afficher
	with st.sidebar:
		st.title('Octroi crédit clients')

		liste_id_clients = fd.recuperer_liste_id_clients(df_demandes_credit)
		id_client = st.selectbox('ID client', liste_id_clients)

		liste_variables = fd.recuperer_liste_variables(df_demandes_credit)
		variable1 = st.selectbox('Feature 1 à afficher', liste_variables, index=0, key="feature1")
		variable2 = st.selectbox('Feature 2 à afficher', liste_variables, index=1, key="feature2")

	# Affichage des visualisations si l'on reçoit bien les prédictions du client
	donnees_client = fd.recuperer_donnees_std_client(id_client, df_std_demandes_credit)
	predictions_client = fd.recuperer_prediction_client(donnees_client)
	if predictions_client:
		st.header('Client #{}'.format(id_client))
		problemes_remboursement = predictions_client['problemes_remboursement']
		score_remboursement_client = predictions_client['score_remboursement_client']

		# Indication sur l'octroi du crédit
		if problemes_remboursement:
			st.info('Demande de crédit refusée')
		else:
			st.info('Demande de crédit acceptée')

		# Visualisation du score sous forme de jauge
		jauge_score = fd.construire_jauge_score(score_remboursement_client)
		st.plotly_chart(jauge_score)

		# Distribution de la première feature choisie par l'utilisateur
		graphique1 = fd.construire_graphique(df_demandes_credit, variable1)
		donnee1_client = fd.recuperer_donnee_client(df_demandes_credit, id_client, variable1)
		graphique1 = fd.ajouter_position_client(graphique1, donnee1_client)		
		st.plotly_chart(graphique1)

		# Distribution de la seconde feature choisie par l'utilisateur
		graphique2 = fd.construire_graphique(df_demandes_credit, variable2)
		donnee2_client = fd.recuperer_donnee_client(df_demandes_credit, id_client, variable2)
		graphique2 = fd.ajouter_position_client(graphique2, donnee2_client)	
		st.plotly_chart(graphique2)

if __name__ == "__main__":
	main()