import random
from typing import Tuple, Dict, List, Any, Union, Optional
from datetime import datetime, timedelta
from datetime import timedelta
from collections import defaultdict
import pickle
from datetime import datetime, timedelta


from exceptions import TestItemsAlreadyUploaded
from . import db_session
from .task import Task
from .recommendation import Recommendation
from .user import User


def gen_new_recommendations(user_id):
    """Сгенерировать новую порцию рекомендаций для пользователя user_id"""
    # TODO 2 - написать функцию, заглушка для генерации
    ### ЗАГЛУШКА:
    db_sess = db_session.create_session()
    print("[INFO] Generating new tasks...")
    task_ids = [i[0] for i in db_sess.query(Task.task_id).all()]
    for j in range(50):
        task_id = random.choice(task_ids)
        is_solved = j < 20
        result = None
        if is_solved:
            result = bool(random.randint(0, 1))
        db_sess.add(Recommendation(user_id=user_id, task_id=task_id,
                                   is_solved=is_solved, result=result,
                                   datetime=get_fake_datetime(0, 10)))
    db_sess.commit()


def init_database(path: str):
    db_session.global_init_lite(path)


def get_fake_datetime(random_days=5, random_hours=10):
    return datetime.now() - timedelta(days=random.randint(1, random_days) if random_days > 1 else 0,
                                      hours=random.randint(1, random_hours) if random_hours > 1 else 0)


def get_tasks_count() -> int:
    db_sess = db_session.create_session()
    return db_sess.query(Task).count()


def add_tasks_from_pickle(name_pickle_file):
    """Достает список словарей из pickle-файла и добавляет элементы в БД заданий"""

    db_sess = db_session.create_session()
    if get_tasks_count() > 0:
        raise TestItemsAlreadyUploaded("Error upload tasks because count tasks already > 0")

    with open(name_pickle_file, "rb") as f:
        tasks = pickle.load(f)
    print(tasks)
    for i in tasks:
        i.pop('task_id', None)
        db_sess.add(Task(**i))
    db_sess.commit()


def add_test_users():
    """Добавляет рандомных чуваков в БД"""

    db_sess = db_session.create_session()
    if db_sess.query(User).count() > 0:
        raise TestItemsAlreadyUploaded("Error upload test users because count users already > 0")

    test_users = [
        {
            "mail": f"{name}_human@hello.ru",
            "registration_datetime": get_fake_datetime(5, 0)
        } for name in ["andrey", "ivan", "anton"]
    ]

    print(test_users)
    for i in test_users:
        db_sess.add(User(**i))
    db_sess.commit()


def add_test_recommendations():
    """Добавляет рандомные рекомендации в БД"""
    db_sess = db_session.create_session()
    if db_sess.query(Recommendation).count() > 0:
        raise TestItemsAlreadyUploaded("Error upload test recommendations because count recommendations already > 0")

    user_ids = [i[0] for i in db_sess.query(User.user_id).all()]
    task_ids = [i[0] for i in db_sess.query(Task.task_id).all()]
    for user_id in user_ids:
        for j in range(100):
            task_id = random.choice(task_ids)
            is_solved = j < 20
            result = None
            if is_solved:
                result = bool(random.randint(0, 1))
            db_sess.add(Recommendation(user_id=user_id, task_id=task_id,
                                       is_solved=is_solved, result=result,
                                       datetime=get_fake_datetime(0, 10)))

    db_sess.commit()


def get_next_rec_for_user_by_id(user_id):
    db_sess = db_session.create_session()
    query = db_sess.query(Recommendation).filter_by(user_id=user_id).filter_by(is_solved=False)
    query = query.order_by("recommendation_id")

    if query.count() == 0:
        # генерируем новую порцию заданий для пользователя
        # TODO 2 - добавить генерацию новых рекоммендаций для конкретного пользователя
        gen_new_recommendations(user_id)
        return get_next_rec_for_user_by_id(user_id)

    return query.first(), db_sess


def add_new_user(mail):
    """Добавляет нового юзера с заданной почтой, возвращает его id-шник"""
    db_sess = db_session.create_session()
    new_user = User(mail=mail, registration_datetime=datetime.now())
    db_sess.add(new_user)
    db_sess.commit()

    add_starter_pack_tasks_for_user(get_user_id_by_user_mail(mail))


def add_starter_pack_tasks_for_user(user_id):
    """Добавляет стартовый набор задач для нового пользователя с id=user_id"""
    # TODO 2 - написать функцию, добавление стартовых заданий новому пользователю

    ### ЗАГЛУШКА:
    db_sess = db_session.create_session()
    task_ids = [i[0] for i in db_sess.query(Task.task_id).all()]
    for j in range(100):
        task_id = random.choice(task_ids)
        is_solved = j < 20
        result = None
        if is_solved:
            result = bool(random.randint(0, 1))
        db_sess.add(Recommendation(user_id=user_id, task_id=task_id,
                                   is_solved=is_solved, result=result,
                                   datetime=get_fake_datetime(0, 10)))

    db_sess.commit()
    #################################################


def get_user_id_by_user_mail(mail):
    db_sess = db_session.create_session()
    user_id = db_sess.query(User).filter_by(mail=mail)
    if user_id.count() == 0:
        return None
    return user_id.one().user_id


def get_next_rec_for_user_by_mail(mail):
    user_id = get_user_id_by_user_mail(mail)
    if not user_id:
        add_new_user(mail)
        print(f"[INFO] new user with mail: {mail}")
        return get_next_rec_for_user_by_mail(mail)
    print(1, end="")
    return get_next_rec_for_user_by_id(user_id)[0]


def check_answer_by_id(user_id, user_answer):
    """ Проверяет ответ пользователя, возвращает и сохраняет результат (правильность ответа) """
    solved_recommendation, db_sess = get_next_rec_for_user_by_id(user_id)
    right_answer = db_sess.query(Task).get(solved_recommendation.task_id).true_answer
    result = user_answer == right_answer
    solved_recommendation.is_solved = True
    solved_recommendation.result = result
    db_sess.commit()
    return result


def check_answer_by_mail(mail, user_answer):
    user_id = get_user_id_by_user_mail(mail)

    if not user_id:
        raise KeyError(f"Not data for user finded, user_mail={mail}")
    return check_answer_by_id(user_id, user_answer)


def get_next_task_for_user_by_mail(mail):
    db_sess = db_session.create_session()
    rec = get_next_rec_for_user_by_mail(mail)
    return db_sess.query(Task).get(rec.task_id)

# TODO - функция получения истории решения конкретным пользователем
