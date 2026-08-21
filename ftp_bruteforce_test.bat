@echo off
cd /d "%~dp0"

call .venv\Scripts\activate.bat

echo ========================================
echo       FTP BRUTE-FORCE TEST
echo ========================================
echo.
echo Target:   127.0.0.1:2121
echo Username: alice
echo.
echo Starting test...
echo.

python ftp_bruteforce_test.py

echo.
echo ========================================
echo       TEST FINISHED
echo ========================================
pause