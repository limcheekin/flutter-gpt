#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sqlalchemy import inspect
from sqlalchemy.engine import Row
from sqlalchemy.ext.declarative import declarative_base

DeclarativeBase = declarative_base()


class Base(DeclarativeBase):
    __abstract__ = True

    @classmethod
    def to_dict(cls, models):
        # 使用.all()查询的情况
        if isinstance(models, list):
            # 查询结果为空
            if len(models) == 0:
                return []
            # 查询结果不包含所有字段的情况
            if isinstance(models[0], Row):
                return [row._asdict() for row in models]
            # 查询结果包含所有字段的情况
            else:
                return [cls._asdict(model) for model in models]
        # 使用.first()查询的情况
        else:
            if not models:
                return {}
            if isinstance(models, Row):
                return models._asdict()
            return cls._asdict(models)

    def _asdict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
