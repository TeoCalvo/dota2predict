# %%
import streamlit as st
import pandas as pd

from pyspark.sql import functions as F

from lib import utils

page_icon = 'https://i.ibb.co/1255VFn/forge-emote.png'
st.set_page_config(page_title="Dota2 Predict", page_icon=page_icon)

print("Buscando modelo no MLFLOW...")
model = utils.get_model()
print("ok.")

print("Abrindo sessão spark...")
spark = utils.get_spark_session()
print("ok.")

print("Buscando times elegíveis...")
teams = utils.get_teams(spark)
print("ok.")

banner = "https://i.ibb.co/B6LZy1S/dota2predict.png"

st.markdown(
    f'<p align="center"><img src="{banner}" alt="Logo" width="550"></p>',
    unsafe_allow_html=True
)

st.title("App de Predições de Dota2")

st.warning("""
Aviso: Este aplicativo fornece informações sobre predições de partidas de Dota2.
Não garantimos resultados e não nos responsabilizamos por apostas feitas com base nessas informações.

AUC Teste = 0,632
""")

radiant_team_box = st.selectbox("Radiant", options=teams['descTeamName'])
dire_team_box = st.selectbox("Dire", options=teams['descTeamName'])

radiant_team_id = teams[teams['descTeamName']==radiant_team_box]['idTeam'].iloc[0]
dire_team_id = teams[teams['descTeamName']==dire_team_box]['idTeam'].iloc[0]

team_ids = [radiant_team_id, dire_team_id]

try:
    print("Buscando dados no database local...")
    fs_teams = utils.get_teams_database_fs(team_ids)
    print("ok.")

except:
   fs_teams = pd.DataFrame() 

if fs_teams.shape[0] < 2:
    print("Coletando dados do Databricks...")
    fs_teams = utils.get_teams_databricks_fs(spark, team_ids)
    print("ok")

    print("Inserindo dados no database local...")
    utils.insert_teams_database_fs(fs_teams)
    print("ok")

df_predict = utils.prepare_predict(fs_teams, radiant_team_id, dire_team_id)
dire_win, radiant_win = model.predict_proba(df_predict[model.feature_names_in_])[0]

df_winner = pd.DataFrame({
    "Radiant": [radiant_team_box],
    "Radiant Prob.": [radiant_win*100],
    "Dire": [dire_team_box],
    "Dire Prob.": [dire_win*100]
})

col_config = {
    "Radiant": "Radiant Team",
    "Radiant Prob.": st.column_config.NumberColumn(
            "Radiant Win Prob.(%)",
            help="Probabilidade de vitória para o time dos Iluminados",
            format="%.2f",
        ),
    "Dire": "Dire Team",
    "Dire Prob.": st.column_config.NumberColumn(
            "Dire Win Prob.(%)",
            help="Probabilidade de vitória para o time dos Temidos",
            format="%.2f",
        ),}

st.dataframe(df_winner, hide_index=True, column_config=col_config)


text_arch = """
## Arquitetura

Nosso modelo foi treinado no ambiente do Databrick, a partir de um Feature Store com +1.000 caracteríticas.

Todo modelo treinado e logado no MLFlow e selecionado para ser colocado em produto. Assim, garantimos que o app no Streamlit não precise se preocupar com nenhum binário e fazer a gestão do modelo que estão em produção.


<p align="center"><img src="https://i.ibb.co/LNKzmNg/dota-predict-drawio.png" alt="Arquitetura" width="550"></p>
    
"""
st.markdown(text_arch, unsafe_allow_html=True)


text_desenvolvimento = """

## Desenvolvimento

Todo o projeto foi desenvolvido em live. Até o momento do primeiro deploy, foram +40 horas de trabalho ao vivo.

Você pode acessar este conteúdo na nossa Twitch ([twitch.tv/teomewhy](https://twitch.tv/teomewhy)), basta ser assinante.

Consideramos a Área sob Curva ROC para escolha do melhor modelo. Onde o mesmo se encontra (em uma base de teste) com 0,632.


"""

st.markdown(text_desenvolvimento, unsafe_allow_html=True)


twitter = "https://logodownload.org/wp-content/uploads/2014/09/twitter-logo-2-1.png"
linkedin = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
instagram = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/2048px-Instagram_icon.png"
twitch = "https://www.freepnglogos.com/uploads/twitch-app-logo-png-3.png"
github = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

social = f"""
<p align="center">
<a href="https://twitter.com/teoCalvo" style="padding: 0 20px"> <img src="{twitter}" alt="Twitter" width="42"></a>
<a href="https://www.linkedin.com/in/teocalvo/" style="padding: 0 20px"> <img src="{linkedin}" alt="Linkedin" width="42"></a>
<a href="https://www.instagram.com/teo.calvo/" style="padding: 0 20px"> <img src="{instagram}" alt="Instagram" width="42"></a>
<a href="https://www.twitch.tv/teomewhy" style="padding: 0 20px"> <img src="{twitch}" alt="Twitch" width="42"></a>
<a href="https://github.com/teomewhy" style="padding: 0 20px"> <img src="{github}" alt="Github" width="42"></a>
</p>
"""

st.markdown(social, unsafe_allow_html=True)