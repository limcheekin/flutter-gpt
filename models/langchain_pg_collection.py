# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.sql.sqltypes import NullType


from . import Base


class LangchainPgCollection(Base):
    __tablename__ = 'langchain_pg_collection'

    uuid = Column(UUID, primary_key=True)
    name = Column(String)
    cmetadata = Column(JSON)
