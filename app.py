# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:28:12 2020

@author: TBEL972
"""

## chargement des librairies
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib
import pandas as pd
import numpy as np

## chargement du modèle
model = joblib.load('model.pkl')
   
## Préparation prédiction

def predict(sepal_length, sepal_width, petal_length, petal_width):
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        return prediction
def predict(sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return prediction


def mayou():


    # Titre de la page
    st.header('**Classification florale**')
    
    st.write("Ceci est l'aboutissement d'un travail de recherche sur les fleurs d'iris, émanant de l'équipe de recherche du Pr Stevens. Le modèle prédictif présenté ici a obtenu un score de **97% de précision**")
    
    ## Création de la sidebar

    selection = st.sidebar.selectbox("Type de prédiction : ", ("Prédiction en temps réel","Prédiction par lot"))
    
    # Logo de l'entreprise
    from PIL import Image
    logo = Image.open('logo.png')
    
    # Personnalisation de la sidebar
    st.sidebar.image(logo, use_column_width=True)
    
    st.sidebar.info("Cette application est une démonstration, conçue par l'Agence Marketic")
    
    st.sidebar.success('Retrouvez-nous sur http://www.agence-marketic.fr')
    
    # Image de fleur
    image = Image.open('setosa.jpg')
    st.sidebar.image(image, use_column_width=True)
    
    ## Personnalisation page principale
    
    st.subheader("Prédictions du type de fleur selon les critères suivants :")
    
    if selection == "Prédiction en temps réel":
        
        sepal_length = st.slider('Indiquez la longueur sépales en mm', min_value=0.1, max_value=9.8, value=2.5)
        sepal_width	= st.slider('Indiquez la largeur sépales en mm', min_value=0.1, max_value=9.9, value=1.5)
        petal_length = st.slider('Indiquez la longueur pétales en mm', min_value=0.1, max_value=9.9, value=0.5)	
        petal_width = st.slider('Indiquez la largeur pétales en mm', min_value=0.1, max_value=9.9, value=3.5)
        
        ## Prédiction finale 
        resultat = ""
        if st.button("Prediction"):
            resultat = predict(sepal_length, sepal_width, petal_length, petal_width)
            st.success("Ces résultats sont caractéristiques d'une fleur de la famille des **{}**".format(resultat))
        
    ## Traitement par lot
    if selection == 'Prédiction par lot':
        batch = st.file_uploader("Télécharger un fichier au format .csv", type="csv")
        
        if batch is not None:
            df=pd.read_csv(batch)
            prediction = model.predict(df)
            proba = model.predict_proba(df)
            
            #mise en forme
            pred = pd.Series(prediction.reshape(150,))
            concat = pd.concat([df, pred], axis=1)
            concat.columns=['Longueur_sépales', 'Largeur_sépales', 'Longueur_pétales', 'Largeur_pétales', 'prédiction']
            st.write(concat)                      
                         
## Lancement de l'application
if __name__=='__main__':
    mayou()
