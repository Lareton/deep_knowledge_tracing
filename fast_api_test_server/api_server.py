from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel
import random
random.seed(42)

from data import db_manager

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    db_manager.init_database("database.db")
    print("Session started...")


@app.post("/get_task/{mail}")
async def get_next_task_for_user_by_mail(mail: str):
    rec_task_id = db_manager.get_next_task_for_user_by_mail(mail).site_task_id
    return {"task_id": str(rec_task_id)}


@app.post("/check_answer/")
async def check_answer_user(mail: str, answer: str):
    result = db_manager.check_answer_by_mail(mail, answer)
    return {"result": result}
