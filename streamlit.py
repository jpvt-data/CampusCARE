import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config( layout="wide", page_title = "Campus Care", page_icon= '../streamlit/logo.png')

@st.cache_data
def charger_csv(lien):
    return pd.read_csv(lien)

X_final = charger_csv("./donnees/nettoyees/df_encode_ml.csv")

@st.cache_data
def pickl_listes(lien):
    with open(lien, 'rb') as f: 
        villes = pickle.load(f)
        etudes = pickle.load(f)
        sommeil = pickle.load(f)
        alim = pickle.load(f)
    return villes, etudes, sommeil, alim

villes, etudes, sommeil, alim = pickl_listes("./donnees/nettoyees/mes_listes.pkl")

@st.cache_data
def pickl_ml(lien):
    with open(lien, 'rb') as f:
        modeles = pickle.load(f)
    SN = modeles['StandardScaler']
    model = modeles['LogisticRegression']
    return SN, model

SN, model = pickl_ml("./donnees/nettoyees/mes_modeles.pkl")

@st.cache_resource
def encodage_predict(df_a_predire):
    X_num = df_a_predire[['Age', 'Pression Académique', 'CGPA', 'Satisfation Académique', 'Temps études/jour(heures)', 'Stress Financier']]
    X_cat = df_a_predire[['Genre', 'Ville', 'Niveau étude', 'Temps de sommeil', 'Habitudes Alimentaires']]
    X_reserve = df_a_predire[['Pensées Suicidaires', 'Antécédents familiaux mentaux']]

    X_num_SN = pd.DataFrame(SN.transform(X_num), columns=X_num.columns)

    X_cat_dummies = pd.get_dummies(X_cat)
    X_encoded_predire = pd.concat([X_num_SN, X_reserve], axis=1)
    X_encoded_predire = pd.concat([X_encoded_predire, X_cat_dummies], axis=1)

    # DataFrame vide qui a les mêmes colonnes que X_final
    X_predict = pd.DataFrame(columns=X_final.columns)

    # On veut que le DataFrame ait le même nombre de lignes que X_encoded_predire
    X_predict = X_predict.reindex(index=X_encoded_predire.index)
    # On met tous les NaN à False
    X_predict = X_predict.fillna(False)

    # On parcourt chaque colonne de X_encoded_predire
    # Si la colonne est présente dans X_final alors on la garde
    # Sinon, on la met à False
    for column in X_final.columns:
        if column in X_encoded_predire.columns:
            X_predict[column] = X_encoded_predire[column]

    return X_predict

@st.cache_data
def booleen(x):
    if x == 'Oui':
        return True
    elif x == 'Non':
        return False

with st.sidebar:
    st.sidebar.image('./streamlit/CampusCare.webp', use_container_width=True)
    st.markdown(
    """
    <style> 
    [data-testid="stSidebar"]{
        background-image: url(https://i.ibb.co/SRzckFS/sidebar.png);
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


# Utiliser option_menu dans l'application
    page = option_menu(
        menu_title=None,  # Titre du menu
        options=["Accueil", "Calculateur de risque", "Statistiques"]
    )

st.markdown(
    """
    <style> 
    [data-testid="stMain"]{
        background-color:  #E8F5E9;
        background-size: cover;
        background-position: center;
    }
    [data-testid="stHeader"]{
        background-color:  #E8F5E9;
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
if page == "Accueil":
    st.markdown("<h1 style='text-align: center;'>🌟 Campus Care 🌟</h1>", unsafe_allow_html=True)
    st.markdown("""
        # Votre bien-être mental est essentiel!

        Campus Care est une application innovante conçue pour aider les étudiants à identifier les risques potentiels de dépression dès les premiers signes.

        En analysant les données que vous fournissez, nous estimons le pourcentage de probabilité que vous présentiez des symptômes dépressifs et vous offrons des pistes pour agir en faveur de votre santé mentale.

        De plus, pour mieux vous sensibiliser aux facteurs de risque de la dépression et aux statistiques inquiétantes, une page dédiée aux **statistiques** est disponible. Vous y trouverez des informations importantes sur les tendances et les chiffres relatifs à la dépression chez les étudiants, afin de mieux comprendre l'ampleur du problème et prendre des mesures préventives.

        Enfin, si vous avez besoin de soutien ou de ressources supplémentaires, nous vous fournissons ici quelques **contacts d’aide** :

        - **Soutien téléphonique** : Ligne d’aide santé mentale – **080-1234-5678** (disponible 24h/24)
        - **Email** : **aide@campuscareindia.org**
        - **Centre de soutien étudiant** : Campus Care, Bâtiment A, Université de Delhi – **Contactez le 011-2345-6789**

        N'hésitez pas à contacter ces ressources en cas de besoin. Nous sommes ici pour vous aider à prendre soin de vous.
        """)
    st.image('./streamlit/accueil.jpg', use_container_width=True)

if page == "Calculateur de risque":
    st.markdown("<h1 style='text-align: center;'>🧠 Risquez vous la dépression ? 🧠</h1>", unsafe_allow_html=True)

    st.write('Répondez à ces quelques questions et nous vous dirons si vous êtes un sujet à risque : ')

    city = st.selectbox('Dans quelle ville vivez vous ?', villes, index=None, placeholder="Choisissez une ville")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Sexe :', ['Homme', 'Femme'], index = None,  placeholder="Choisissez une option")
        age = st.number_input("Age :", min_value=0, max_value=120, step=1, format="%d")
        niv = st.selectbox("Quel est votre niveau d'étude ? ", etudes, index = None,  placeholder="Choisissez une option")
        sleep = st.selectbox('Quelle est votre temps de sommeil moyen ?', sommeil, index=None, placeholder="Choisissez parmis la sélection :")
        
        

    with col2:
        pression = st.select_slider("Notez la pression académique subie : ", [0,1,2,3,4,5])
        satisf = st.select_slider("Notez votre satisfaction de vos études : ", [0,1,2,3,4,5])
        finance = st.select_slider("Notez le stress financier que vous subissez : ", [0,1,2,3,4,5])
        cgpa = st.slider( "Votre indice CGPA : ", min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
        
        
    with col3:
        work = st.number_input("Nombre d'heures d'études journalières :", min_value=0, max_value=14, step=1, format="%d")
        food = st.selectbox('Comment jugez-vous vos habitudes alimentaires ?', alim, index=None, placeholder="Choisissez une option")
        illness = st.selectbox('Antécédents familiaux de maladies mentales ?', ['Oui', 'Non'], index=None, placeholder="Choisissez une option")
        suicide = st.selectbox('Avez-vous déjà eu des pensées suicidaires ?', ['Oui', 'Non'], index=None, placeholder="Choisissez une option")

    if (city != None) and (gender != None) and (age != 0) and (niv != None) and (sleep != None) and (cgpa != 0) and (food != None) and (illness != None) and (suicide != None):

        dico = {
            'Genre' : gender,
            'Age' : age, 
            'Ville' : city, 
            'Niveau étude' : niv, 
            'Pression Académique' : pression, 
            'CGPA' : cgpa,
            'Satisfation Académique' : satisf,
            'Temps de sommeil' : sleep, 
            'Habitudes Alimentaires' : food,
            'Pensées Suicidaires' : booleen(suicide), 
            'Temps études/jour(heures)' : work, 
            'Stress Financier' : finance,
            'Antécédents familiaux mentaux' : booleen(illness)
        }

        df_predict = pd.DataFrame([dico])
        predict = encodage_predict(df_predict)
        proba = float(np.round(model.predict_proba(predict)[0, 1] * 100, 2))

        st.markdown("<h2 style='text-align: center;'>Probabilité que vous soyez en dépression : </h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{proba} %</h3>", unsafe_allow_html=True)

        if proba > 45:
            st.markdown(f"<h3 style='text-align: center;'>N'hésitez pas : Consultez rapidement votre médecin</h3>", unsafe_allow_html=True)

if page == "Statistiques":
    st.markdown("<h1 style='text-align: center;'>📉 Plongée dans les chiffres 📈</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Comprendre votre santé mentale</h1>", unsafe_allow_html=True)
    st.markdown("""
        Bienvenue dans notre espace dédié aux statistiques sur la santé mentale des étudiants ! 
                
        Ici, nous vous invitons à explorer les données et à prendre conscience de l'impact des différents facteurs sur votre bien-être mental. Vous découvrirez des informations clés, des tendances et des corrélations qui mettent en lumière les défis auxquels font face de nombreux étudiants.

        À travers des visualisations et des indicateurs clés, vous pourrez observer les comportements, les habitudes de vie et les pressions académiques qui peuvent influencer la santé mentale. Nos statistiques ne sont pas seulement des chiffres, mais une porte d'entrée vers une meilleure compréhension de soi et des ressources disponibles pour prendre soin de votre bien-être.

        Explorez, comprenez et surtout, prenez des décisions éclairées pour votre santé mentale !
            """)
    
    col1, col2, col3 = st.columns([2, 10, 2])
    with col2:
        st.image('.\images\dashboard.png', use_container_width=True)
    


    col1, col2 = st.columns(2)

    with col1:
        st.image('./streamlit/depressifs.png')

    with col2:
        
        st.markdown("<h3 style='text-align: center;'>Proportion d'étudiants dépressifs</h3>", unsafe_allow_html=True)
        st.markdown("""
        Ce graphique montre le pourcentage d'étudiants qui vivent avec une dépression. Cela nous aide à comprendre combien d'entre nous sont touchés par cette réalité.
                    
        **Pourquoi cela vous concerne ?**
                    
        La dépression touche plus de la moitié des étudiants. Si vous vous sentez dépassé, sachez que vous n'êtes pas seul. Reconnaître les signes de la dépression et chercher de l'aide tôt peut faire une grande différence. 
                    
        Ce diagramme est un rappel que prendre soin de votre santé mentale est aussi important que vos études.
            """)
        

        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Répartition des heures de sommeil</h3>", unsafe_allow_html=True)
        st.markdown("""
            Ce graphique vous montre combien d’heures de sommeil en moyenne les étudiants accumulent chaque nuit. Vous pourrez voir si vous dormez suffisamment ou si des ajustements sont nécessaires.

            **Pourquoi cela vous concerne ?**
            Le sommeil joue un rôle fondamental dans votre bien-être mental et physique. Si vous êtes parmi ceux qui dorment moins de 5 heures, sachez que cela peut augmenter votre stress et affecter vos émotions. 
                    
            Essayez de prioriser une bonne nuit de sommeil : c'est un moyen simple de prendre soin de vous et d'améliorer vos performances académiques et personnelles.
                        """)        


    with col2:
        st.image('./streamlit/sommeil.png')



    col1, col2 = st.columns(2)

    with col1:
        st.image('./streamlit/alimentation.png')


    with col2:
        
        st.markdown("<h3 style='text-align: center;'>Répartition des habitudes alimentaires</h3>", unsafe_allow_html=True)
        st.markdown("""
        Ce graphique vous montre les types d'aliments que les étudiants consomment en majorité: sains, modérés ou mauvais pour la santé. Il reflète l'impact de notre alimentation sur notre bien-être.

        **Pourquoi cela vous concerne ?**
                    
        Votre alimentation influence directement votre humeur et votre énergie. Si vous avez tendance à consommer des aliments peu équilibrés, cela peut affecter votre concentration et votre humeur. 
                    
        Adopter des habitudes alimentaires plus saines peut réduire le stress et vous aider à vous sentir mieux au quotidien. Votre bien-être passe aussi par votre assiette !
                    
        """)
                    

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Niveau moyen de pression académique</h3>", unsafe_allow_html=True)
        st.markdown("""
            Ce graphique montre le niveau moyen de stress académique que les étudiants ressentent. Il nous aide à comprendre si nous sommes trop sollicités par nos études.

            **Pourquoi cela vous concerne ?**
                    
            Le stress académique est un problème que beaucoup d'entre nous connaissent. Si vous ressentez de la pression, sachez que vous n'êtes pas seul. Apprendre à gérer ce stress et à organiser vos priorités est essentiel. 
                    
            En prenant du recul et en vous entourant des bonnes ressources, vous pouvez mieux supporter cette pression et préserver votre bien-être.
                        """)        


    with col2:
        st.image('./streamlit/pression.png')

    col1, col2 = st.columns(2)

    with col1:
        st.image('./streamlit/suicide.png')



    with col2:
        
        st.markdown("<h3 style='text-align: center;'>Proportion d'étudiants avec pensées suicidaires</h3>", unsafe_allow_html=True)
        st.markdown("""
        Ce graphique montre le pourcentage d'étudiants qui ont eu des pensées suicidaires. C’est un indicateur important pour comprendre à quel point certains d’entre nous peuvent se sentir perdus ou désemparés.

        **Pourquoi cela vous concerne ?**
                    
        Si vous vous sentez dans une situation de désespoir ou que vous avez des pensées suicidaires, il est crucial de demander de l'aide. Plus de 60 % des étudiants ont rapporté ce genre de pensées. N'ayez pas peur de parler à quelqu'un de confiance. 
                    
        Vous méritez de vous sentir soutenu et d’avoir les ressources nécessaires pour aller mieux.
                    
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Répartition géographique des étudiants dépressifs</h3>", unsafe_allow_html=True)
        st.markdown("""
            Ce graphique vous montre où les étudiants souffrant de dépression sont les plus nombreux, selon les régions. Il vous permet de mieux comprendre les disparités géographiques.

            **Pourquoi cela vous concerne ?**
                    
            Certaines régions sont plus touchées que d’autres par la dépression. Si vous êtes dans une zone où la dépression est plus présente, il est d’autant plus important de chercher du soutien et de prendre soin de vous. 
                    
            Peu importe où vous êtes, des ressources sont disponibles pour vous aider à surmonter les moments difficiles. N'attendez pas pour agir et prendre soin de votre santé mentale.
                        """)        


    with col2:
        st.image('./streamlit/map.png')


        

        
        