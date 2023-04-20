# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.sql.sqltypes import NullType


from . import Base


class Channels(Base):
    __tablename__ = 'channels'

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    name = Column(String(255), nullable=False, unique=True)
    cmetadata = Column(JSON)
