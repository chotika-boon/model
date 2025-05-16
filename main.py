import pandas as pd
from google.cloud import bigquery
from card_scoring_model import CardRecommender
from datetime import datetime

def run_model_and_upload(request):
    # ใช้ URL ที่เปิด public แล้ว
    df = pd.read_csv('https://storage.googleapis.com/coolkid-data/cards.csv')

    model = CardRecommender()
    scored_df = model.score_cards(df)

    client = bigquery.Client()
    table_id = "coolkid-460014.card_scoring.daily_card_ranking"
    
    job = client.load_table_from_dataframe(scored_df[[
        'store', 'card_name', 'benefit_detail',
        'bonus_score', 'embedding_score',
        'final_score', 'rank_in_store', 'scored_date'
    ]], table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"))

    job.result()
    return "✅ Card scoring completed and uploaded to BigQuery"
