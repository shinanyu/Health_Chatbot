from sqlalchemy import Column, Integer, String, Date, Enum, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()
class GenderEnum(enum.Enum):
    M = "M"
    F = "F"

class Member(Base):
    __tablename__ = "member_tb"  

    member_no = Column(Integer, primary_key=True, index=True) 
    email = Column(String(256), nullable=False)
    password = Column(String, nullable=False)
    gender = Column(Enum(GenderEnum), nullable=False) 
    birth_date = Column(Date, nullable=False)
    name = Column(String(64), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)
