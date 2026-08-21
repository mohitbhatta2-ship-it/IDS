@echo off
cd /d "%~dp0"

call .venv\Scripts\activate.bat

if not exist "ftp_data" mkdir "ftp_data"

echo ========================================
echo        IDS FTP SERVER
echo ========================================
echo.
echo Host:     127.0.0.1
echo Port:     2121
echo Username: alice
echo Password: correcthorse
echo Folder:   %CD%\ftp_data
echo.
echo FTP server is running...
echo Press Ctrl+C to stop.
echo ========================================
echo.

python -c "from pyftpdlib.authorizers import DummyAuthorizer; from pyftpdlib.handlers import FTPHandler; from pyftpdlib.servers import FTPServer; import os; a=DummyAuthorizer(); a.add_user('alice','correcthorse',os.path.abspath('ftp_data'),perm='elradfmw'); h=FTPHandler; h.authorizer=a; FTPServer(('127.0.0.1',2121),h).serve_forever()"

pause