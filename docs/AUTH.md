# Auth (v5) — Basic Auth Middleware

## Обзор

Опциональная HTTP Basic Auth защита для endpoints:
- `/chat`, `/query` —问答
- `/ingest`, `/documents`, `/documents/reindex` — админка
- `/evaluation/*` — оценка качества
- `/ab/test`, `/calibrate` — A/B тестирование
- `/samples/ingest`, `/bm25/rebuild` — утилиты

**Публичные** (без auth):
- `/health` — проверка здоровья
- `/metrics` — Prometheus endpoint
- `/docs`, `/openapi.json`, `/redoc` — Swagger UI

## Включение

В `.env`:
```bash
AUTH_ENABLED=True
AUTH_USER=admin
AUTH_PASSWORD=your_strong_password_here
AUTH_PUBLIC_PATHS=/health,/metrics,/docs,/openapi.json,/redoc
```

Перезапуск:
```bash
docker compose restart rag-backend
```

## Использование

### curl
```bash
# Без auth → 401
curl http://localhost:8000/chat

# С auth
curl -u admin:your_password http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Сколько дней отпуск?"}'
```

### Python (requests)
```python
import requests
resp = requests.post(
    "http://localhost:8000/chat",
    json={"question": "Сколько дней отпуск?"},
    auth=("admin", "your_password"),
)
```

### Python (httpx)
```python
import httpx
client = httpx.Client(auth=("admin", "your_password"))
resp = client.post("http://localhost:8000/chat", json={"question": "..."})
```

### Gradio UI
Gradio UI контейнер (`rag-ui`) ходит в backend без auth (внутренняя docker-сеть).
Если auth включён — Gradio UI не сможет подключиться. Решения:

1. **Не включать auth в демо-режиме** (контейнер в изолированной сети)
2. **Прокси с auth перед UI** (nginx + basic auth)
3. **Добавить auth в `app/ui.py`** (патч `BACKEND_URL` с auth в URL: `http://admin:pass@rag-backend:8000`)

## Безопасность

### Timing-safe сравнение
Используется `secrets.compare_digest()` — защита от timing-атак.

### HTTPS
Basic Auth передаёт credentials в Base64 — **нельзя использовать без HTTPS** в production.
Рекомендуется поставить обратный прокси (nginx/caddy) с TLS.

### Смена пароля
1. Сгенерировать надёжный пароль (минимум 16 символов):
   ```bash
   openssl rand -base64 24
   ```
2. Установить в `.env`:
   ```
   AUTH_PASSWORD=<сгенерированный_пароль>
   ```
3. Перезапустить: `docker compose restart rag-backend`

### Логи
Неудачные попытки авторизации логируются:
```
WARNING Auth failed for user='admin' from 192.168.1.42
```

## Проверка

```bash
# Должен вернуть 200 с auth
curl -u admin:your_password http://localhost:8000/health
# (health публичный, но auth не помешает)

# Должен вернуть 401 без auth
curl http://localhost:8000/chat

# Должен вернуть 200 с auth
curl -u admin:your_password -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

## Отключение

```bash
AUTH_ENABLED=False
docker compose restart rag-backend
```
