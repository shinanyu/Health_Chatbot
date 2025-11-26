from sqlalchemy import Column, Integer, Float, Date, ForeignKey, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DailyActivity(Base):
    __tablename__ = "daily_activities_tb" 

    id = Column(Integer, primary_key=True, index=True)
    member_no = Column(Integer, ForeignKey("member_tb.member_no"), nullable=False)
    date = Column(Date)
    sleep_hours = Column(Float)
    caffeine_mg = Column(Float)
    alcohol_consumption = Column(Float)
    physical_activity_hours = Column(Float)
    predicted_sleep_quality = Column(Float)
    predicted_fatigue_score = Column(Float)
    recommended_sleep_range = Column(String)
    condition_level = Column(String)
