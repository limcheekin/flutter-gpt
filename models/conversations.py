# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.sql.sqltypes import NullType


from . import Base

from .channels import Channels
from .users import Users




class Conversations(Base):
    __tablename__ = 'conversations'

    id = Column(BigInteger, primary_key=True, server_default=FetchedValue())
    user_id = Column(ForeignKey('users.id'), nullable=False, index=True)
    channel_id = Column(ForeignKey('channels.id'), nullable=False, index=True)
    context = Column(Text, nullable=False)
    cmetadata = Column(JSON)
    created_at = Column(DateTime, server_default=FetchedValue())

    channel = relationship('Channels', primaryjoin='Conversations.channel_id == Channels.id', backref='conversationss')
    user = relationship('Users', primaryjoin='Conversations.user_id == Users.id', backref='conversationss')
