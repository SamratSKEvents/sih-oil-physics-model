@echo off
cd /d "%~dp0"
if not exist "node_modules" call npm install
start "Slick Lab" /D "%~dp0" cmd /k "npm run dev"
timeout /t 3 >nul
start "" http://localhost:5288/
