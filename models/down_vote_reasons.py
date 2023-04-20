# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.sql.sqltypes import NullType


from . import Base


class DownVoteReasons(Base):
    __tablename__ = 'down_vote_reasons'

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    description = Column(String(255), nullable=False, unique=True)
