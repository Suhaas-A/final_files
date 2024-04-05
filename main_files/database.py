from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine('sqlite:///BlogMan.db', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)


class Users(Base):
    __tablename__ = 'Users'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    username = Column('username', String, unique=True, nullable=False)
    password = Column('password', String, nullable=False)
    email = Column('email', String, unique=True, nullable=False)
    create_date = Column('create_date', DateTime, nullable=False)


Base.metadata.create_all(engine)
