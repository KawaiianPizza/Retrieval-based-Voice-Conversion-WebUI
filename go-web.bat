cd /d %~dp0
call .venv\Scripts\activate.bat
.venv\Scripts\python.exe infer-web.py --pycmd .venv\Scripts\python.exe --port 7897
pause
