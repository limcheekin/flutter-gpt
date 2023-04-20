# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.sql.sqltypes import NullType


from . import Base

from .down_vote_reasons import DownVoteReasons
from .models import Models
from .conversations import Conversations
from .users import Users




class Messages(Base):
    __tablename__ = 'messages'

    id = Column(BigInteger, primary_key=True, server_default=FetchedValue())
    conversation_id = Column(ForeignKey('conversations.id'), nullable=False, index=True)
    user_id = Column(ForeignKey('users.id'), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    bot_message = Column(Text, nullable=False)
    vote = Column(SmallInteger, server_default=FetchedValue())
    down_vote_reason_id = Column(ForeignKey('down_vote_reasons.id'))
    user_feedback = Column(Text)
    model_id = Column(ForeignKey('models.id'), nullable=False)
    cmetadata = Column(JSON)
    created_at = Column(DateTime, server_default=FetchedValue())

    conversation = relationship('Conversations', primaryjoin='Messages.conversation_id == Conversations.id', backref='messagess')
    down_vote_reason = relationship('DownVoteReasons', primaryjoin='Messages.down_vote_reason_id == DownVoteReasons.id', backref='messagess')
    model = relationship('Models', primaryjoin='Messages.model_id == Models.id', backref='messagess')
    user = relationship('Users', primaryjoin='Messages.user_id == Users.id', backref='messagess')
