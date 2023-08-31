import mlflow
from pyspark.sql import functions as F
import pandas as pd
import sqlalchemy
import datetime
from databricks.connect import DatabricksSession

import streamlit as st



@st.cache_resource(ttl='1d')
def get_model(stage="production"):
    mlflow.set_registry_uri("databricks")
    model = mlflow.sklearn.load_model(f"models:/dota_pre_match/{stage}")
    return model

@st.cache_resource(ttl='1h')
def get_spark_session():
    
    # spark = DatabricksSession.builder.profile("DEFAULT").getOrCreate()

    DATABRICKS_HOST = st.secrets['DATABRICKS_HOST']
    DATABRICKS_TOKEN = st.secrets['DATABRICKS_TOKEN']
    DATABRICKS_CLUSTER = st.secrets['DATABRICKS_CLUSTER']

    spark = (DatabricksSession.builder
                              .host(DATABRICKS_HOST)
                              .token(DATABRICKS_TOKEN)
                              .clusterId(DATABRICKS_CLUSTER)
                              .getOrCreate())
    
    return spark

@st.cache_resource(ttl='1d')
def get_teams(_spark):
    teams = (_spark.table('silver.dota.team_last_seen')
                  .select('idTeam','descTeamName')
                  .toPandas()
                  .sort_values('descTeamName'))
    return teams

def get_teams_databricks_fs(spark, team_ids:list):
    
    max_date = (spark.table("feature_store.dota_teams_0")
                     .select(F.max("dtReference").alias('dt'))
                     .toPandas()['dt'].iloc[0])

    teams = ",".join([f"'{i}'" for i in team_ids])
    str_filter = f"dtReference='{max_date}' AND idTeam IN ({teams})"

    fs_teams = (spark.table("feature_store.dota_teams_0")
                     .filter(str_filter)
                     .drop("idTeamRadiant")
                     .drop("descTeamNameRadiant")
                     .drop("descTeamTagRadiant")
                     .drop("idTeamDire")
                     .drop("descTeamNameDire")
                     .drop("descTeamTagDire")
                     .drop("dtReference")
                     .toPandas())
    
    return fs_teams

def get_teams_database_fs(team_ids:list):
    
    dt_today = datetime.datetime.now().strftime("%Y-%m-%d")

    con = sqlalchemy.create_engine("sqlite:///data/database.db")
    teams = ",".join([f"'{i}'" for i in team_ids])
    query = f"""
    SELECT *
    FROM dota_teams
    WHERE dtReference ='{dt_today}'
    AND idTeam IN ({teams})
    """

    return pd.read_sql_query(query, con)


def insert_teams_database_fs(df:pd.DataFrame):
    dt_today = datetime.datetime.now().strftime("%Y-%m-%d")
    con = sqlalchemy.create_engine("sqlite:///data/database.db")

    teams = ",".join([f"'{i}'" for i in df['idTeam']] )
    query = f"DELETE FROM dota_teams WHERE IdTeam IN ({teams});"
    
    try:
        with con.connect() as connect:
            connect.execute(statement=sqlalchemy.text(query))
            connect.commit()
    except:
        print("Falha ao tentar deletar linhas")

    df['dtReference'] = dt_today
    df.to_sql("dota_teams", con=con, index=False, if_exists="append")


def prepare_predict(fs_teams, radiant_team_id, dire_team_id):
    
    radiant_features = (fs_teams[fs_teams['idTeam']==radiant_team_id]
                            .rename(columns={i: f"{i}Radiant" for i in fs_teams.columns})
                            .reset_index(drop=True)
                    )
    
    dire_features = (fs_teams[fs_teams['idTeam']==dire_team_id]
                        .rename(columns={i: f"{i}Dire" for i in fs_teams.columns})
                        .reset_index(drop=True)                        
                        )


    df_predict = pd.concat([radiant_features, dire_features], axis=1)

    return df_predict

