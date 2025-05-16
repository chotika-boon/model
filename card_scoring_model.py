
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

class CardRecommender:
    def __init__(self):
        self.ideal_benefits = [
            "ส่วนลดสูงสุด",
            "เครดิตเงินคืนทันที",
            "ฟรีเมนูพิเศษ",
            "สิทธิ์ที่คุ้มค่าสำหรับร้านอาหาร"
        ]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.ideal_benefits)
        self.ideal_vectors = self.vectorizer.transform(self.ideal_benefits)

    def extract_bonus_score(self, text):
        patterns = [
            (r'ส่วนลด\s*(\d+)\s*%', 0.1),
            (r'(ฟรี|แถม)\s*(\d+)', 1.0),
            (r'คะแนนสะสม\s*(\d+)', 0.1),
            (r'แลกคะแนน\s*(\d+)', 1 / 1000),
            (r'(มูลค่า|ราคา)[^\d]*(\d+)', 1 / 100),
        ]
        for pattern, scale in patterns:
            match = re.search(pattern, text)
            if match and 'เมื่อใช้จ่ายครบ' not in text:
                return int(match.group(1)) * scale
        return 0.0

    def score_cards(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        benefit_vectors = self.vectorizer.transform(df['benefit_detail'].astype(str))
        df['embedding_score'] = cosine_similarity(benefit_vectors, self.ideal_vectors).mean(axis=1)
        df['bonus_score'] = df['benefit_detail'].apply(self.extract_bonus_score)
        df['final_score'] = df['bonus_score'] * 0.75 + df['embedding_score'] * 0.25
        df['rank_in_store'] = df.groupby('store')['final_score'].rank(ascending=False, method='min')
        df['scored_date'] = date.today()
        return df
