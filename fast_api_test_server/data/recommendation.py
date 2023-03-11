from sqlalchemy import Column, ForeignKey, INT, TEXT, BOOLEAN, CHAR, DATETIME
from .db_session import SqlAlchemyBase


class Recommendation(SqlAlchemyBase):
    __tablename__ = 'Recommendations'
    recommendation_id = Column(INT, unique=True, autoincrement=True, primary_key=True)
    user_id = Column(INT, ForeignKey('Users.user_id'), nullable=False, index=True)
    task_id = Column(INT, ForeignKey('Tasks.task_id'), nullable=False)
    is_solved = Column(BOOLEAN, nullable=False)
    result = Column(BOOLEAN, nullable=True)
    datetime = Column(DATETIME, nullable=False)

    def __repr__(self):
        return f'<Recommendation with id={self.recommendation_id}; ' \
               f'user_id={self.user_id}>; ' \
               f'task_id={self.task_id}; ' \
               f'is_solved={self.is_solved}; ' \
               f'result={self.result};>'
