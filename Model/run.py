import subprocess
import time
import webbrowser
import socket

HOST = "127.0.0.1"
PORT = 8000

def is_server_running():
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except:
        return False

print("🚀 تشغيل السيرفر...")

# تشغيل السيرفر
process = subprocess.Popen([
    "/home/ahmed/tf_gpu/bin/python",
    "-m",
    "uvicorn",
    "main:app",
    "--reload"
])

# الانتظار حتى يشتغل السيرفر
while not is_server_running():
    time.sleep(0.5)

print("🌐 فتح المتصفح...")

webbrowser.open(f"http://{HOST}:{PORT}")

# إبقاء البرنامج شغال
process.wait()
