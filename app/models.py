from sqlalchemy import Column, Integer, String, LargeBinary
from app.database import Base

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    embedding = Column(LargeBinary)  # Serialized numpy array
