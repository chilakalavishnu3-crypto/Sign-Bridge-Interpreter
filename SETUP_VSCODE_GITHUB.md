# SignBridge - VS Code / GitHub Setup

## 1. Open in VS Code
Open this folder (the folder containing `backend`, `app`, `assets`, and `requirements.txt`).

## 2. Python
Use Python 3.11.9.

## 3. Create/activate virtual environment (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

## 5. Run
```powershell
python -m uvicorn backend.server:app --reload
```
Open http://127.0.0.1:8080

Or double-click `run.bat`.

## 6. GitHub
```powershell
git init
git branch -M main
git add .
git commit -m "Initial SignBridge project"
git remote add origin https://github.com/vishnu123/signbridge.git
git push -u origin main
```
If `origin` already exists, use:
```powershell
git remote set-url origin https://github.com/vishnu123/signbridge.git
```

Do not commit `.env`, `.venv`, or API keys.
