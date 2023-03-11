from sqlalchemy import Column, INT, TEXT
from .db_session import SqlAlchemyBase


class Task(SqlAlchemyBase):
    __tablename__ = 'Tasks'
    task_id = Column(INT, unique=True, autoincrement=True, primary_key=True)
    site_task_id = Column(INT, index=True, nullable=False)
    true_answer = Column(TEXT, nullable=False)

    def __repr__(self):
        return f'<Task with id={self.task_id}>; ' \
               f'true_answer={self.true_answer};>'