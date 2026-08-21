@echo off
cd /d "%~dp0"

call .venv\Scripts\activate.bat

cd webapp_django

start "" http://127.0.0.1:8000/

python manage.py runserver

pause
