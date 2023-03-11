from sqlalchemy import Column, ForeignKey, INT, TEXT, BOOLEAN, CHAR, DATETIME
from .db_session import SqlAlchemyBase


class User(SqlAlchemyBase):
    __tablename__ = 'Users'
    user_id = Column(INT, unique=True, autoincrement=True, primary_key=True)
    mail = Column(TEXT, nullable=True, index=False)
    registration_datetime = Column(DATETIME, nullable=False)

    def __repr__(self):
        return f'<User id={self.user_id}; ' \
               f'with mail: {self.mail}; ' \
               f'sign up: {self.registration_datetime};>'
