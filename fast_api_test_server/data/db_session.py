import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.orm import Session
import sqlalchemy.ext.declarative as dec
from sqlalchemy.engine import Engine
from sqlalchemy import event

SqlAlchemyBase = dec.declarative_base()

__factory = None



@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=MEMORY")
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.close()


def global_init_lite(db_file):
    global __factory

    if __factory:
        return

    db_file = db_file.strip()
    if not db_file:
        raise Exception("Необходимо указать файл базы данных.")

    conn_str = f'sqlite:///{db_file}?check_same_thread=False'
    print(f"Подключение к базе данных по адресу {conn_str}")

    engine = sa.create_engine(conn_str, echo=False)
    __factory = orm.sessionmaker(bind=engine)

    from . import __all_models
    SqlAlchemyBase.metadata.create_all(engine)


def global_init_post(db_file):
    global __factory

    if __factory:
        return

    db_file = db_file.strip()
    if not db_file:
        raise Exception("Необходимо указать файл базы данных.")

    conn_str = f"mysql+pymysql://root:arduino14@localhost/brainster"
    print(f"Подключение к базе данных по адресу {conn_str}")

    engine = sa.create_engine(conn_str, echo=False)
    __factory = orm.sessionmaker(bind=engine)

    from . import __all_models
    SqlAlchemyBase.metadata.create_all(engine)


def create_session() -> Session:
    global __factory
    if __factory is not None:
        return __factory()
    raise ValueError('You need to init db first. '
                     'Use global_init("name.db")')