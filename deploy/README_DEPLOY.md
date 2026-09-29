# Развёртывание DXA QC

Публичный сервер: Ubuntu, Docker Engine с Compose plugin, Nginx. Код размещается в `/opt/dxa-qc`. Проверенные локальные веса передаются отдельно в каталог `/opt/dxa-qc-assets` в соответствии с их лицензиями; он не находится в Git и монтируется только для чтения. До сборки установите `DXA_ASSETS_DIR=/opt/dxa-qc-assets` и проверьте SHA256 по локальному инвентарю.

Запуск: `bash deploy/check_release.sh`. Если образ уже собран и проверен вручную, `DXA_SKIP_BUILD=1 bash deploy/check_release.sh` использует его без повторной сборки. Compose публикует порт приложения только на `127.0.0.1:8000`. Для GPU установите NVIDIA Container Toolkit и запустите `docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml up -d --build` с `DXA_QC_DEVICE=cuda`. Без GPU используйте `DXA_QC_DEVICE=cpu docker compose -f deploy/docker-compose.yml up -d --build`. Производительность CPU нужно измерить на целевом сервере.

Nginx проксирует `:80` на `127.0.0.1:8000`. Минимальный site config:

```nginx
server {
    listen 80;
    server_name _;
    client_max_body_size 520m;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_read_timeout 360s;
        proxy_send_timeout 360s;
    }
}
```

Проверьте `nginx -t`, затем перезагрузите Nginx и проверьте `/api/health` извне. Uploads хранятся в tmpfs контейнера и удаляются после TTL или перезапуска. Не загружайте идентифицируемые медицинские данные на публичный стенд.
