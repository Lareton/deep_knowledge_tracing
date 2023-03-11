import random, time

from data import db_manager
from exceptions import TestItemsAlreadyUploaded

random.seed(42)

db_manager.init_database("database.db")
print("Session started...")

try:
    db_manager.add_tasks_from_pickle("tasks.pickle")
except TestItemsAlreadyUploaded:
    print("[WARN] tasks already uploaded")

try:
    db_manager.add_test_users()
except TestItemsAlreadyUploaded:
    print("[WARN] users already uploaded")

try:
    db_manager.add_test_recommendations()
except TestItemsAlreadyUploaded:
    print("[WARN] recommendations already uploaded")

for user_id in range(1, 4):
    print(f"rec for user {user_id}:", db_manager.get_next_rec_for_user_by_id(user_id))

mail = "andrey_human@hello.ru"
print(f"rec for {mail}: ", db_manager.get_next_rec_for_user_by_mail(mail))

mail = "me@hello.ru"
time_start = time.time()

for i in range(128):
    print(f"{i} rec for {mail}: ", db_manager.get_next_rec_for_user_by_mail(mail))
    user_answer = str(random.randint(0, 1))
    print("result user: ", db_manager.check_answer_by_mail(mail, user_answer))
    print()

print("time executed: ", time.time() - time_start)
print("\nSession finished...")

