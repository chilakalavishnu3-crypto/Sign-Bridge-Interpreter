@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Creating Python 3.11 virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (echo Failed to create .venv. & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. & exit /b 1)
python -m uvicorn backend.server:app --reload
