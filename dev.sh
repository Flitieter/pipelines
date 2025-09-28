PORT="${PORT:-10086}"
python -m uvicorn main:app --port $PORT --host 127.0.0.1 --forwarded-allow-ips '*' --reload