from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = "sqlite:////data/velovision.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    camera_id = Column(String, index=True)
    image_path = Column(String)
    analysis_text = Column(Text)
    faces_detected = Column(String) # Comma separated names
    prompt_used = Column(Text)
    is_reviewed = Column(Boolean, default=False)

class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    role = Column(String) # Owner, Resident, Visitor
    category = Column(String, default="Uncategorized") # Family, Courier, Neighbor, etc.
    encoding = Column(LargeBinary) # Pickled encoding
    image_path = Column(String)
    last_seen = Column(DateTime)
    sighting_count = Column(Integer, default=0)

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, index=True) # Linked to Event.id
    recipient_value = Column(String, index=True) # The phone/id
    recipient_name = Column(String) # Name displayed in UI
    status = Column(String) # "success", "failed"
    timestamp = Column(DateTime, default=datetime.now)

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
