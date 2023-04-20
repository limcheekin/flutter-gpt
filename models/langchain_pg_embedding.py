# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.sql.sqltypes import NullType


from . import Base

from .langchain_pg_collection import LangchainPgCollection




class LangchainPgEmbedding(Base):
    __tablename__ = 'langchain_pg_embedding'

    uuid = Column(UUID, primary_key=True)
    collection_id = Column(ForeignKey('langchain_pg_collection.uuid', ondelete='CASCADE'))
    embedding = Column(NullType)
    document = Column(String)
    cmetadata = Column(JSON)
    custom_id = Column(String)

    collection = relationship('LangchainPgCollection', primaryjoin='LangchainPgEmbedding.collection_id == LangchainPgCollection.uuid', backref='langchain_pg_embeddings')
