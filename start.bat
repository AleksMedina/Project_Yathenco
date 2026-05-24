@echo off
cd /d "C:\Users\User\PycharmProjects\pythonProject"
call .venv\Scripts\activate
cd /d "C:\Users\User\PycharmProjects\pythonProject\face_recognization"
python -m uvicorn bacend:app --host 0.0.0.0 --port 8000 --reload
pause