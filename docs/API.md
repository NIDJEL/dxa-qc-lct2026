# API и контракты

## Batch CLI

`python scripts/run_dxa_qc_service.py INPUT --device auto --csv OUT.csv --json OUT.json [--study-groups MAP.json]`

Основной CSV имеет ровно поля `path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing`. Область: `Поясничный отдел позвоночника` или `Проксимальный отдел бедра`. Несколько нарушений соединяются `;`; отсутствие отмеченных нарушений — пустая строка. `quality_class=0` означает оценку качества без отметки нарушения, `1` — quality-head отметила нарушение. Это модельный вывод, не экспертная истина.

`--study-groups` принимает `dxa_study_groups_v1`: массив `studies` из `{source_study_id, files}` с точными относительными путями. Карта обязательна только когда исходную границу нельзя однозначно вывести из UID. Нечитаемые файлы и ошибки группировки сохраняются в JSON. Политика CSV Failure/unknown ожидает подтверждения организаторов.

## Web API

- `GET /api/health` — состояние процесса.
- `POST /api/jobs` — multipart DICOM upload, создаёт задание.
- `GET /api/jobs/{id}` — состояние и прогресс.
- `GET /api/jobs/{id}/results` — структурированные результаты.
- `GET /api/jobs/{id}/export.csv` и `/export.json` — экспорт.
- `DELETE /api/jobs/{id}` — удаление задания.

Ограничения: 512 файлов, 512 MiB на задание, 128 MiB на файл; одно тяжёлое inference-задание одновременно. Загрузки удаляются по TTL или DELETE. Публичный стенд принимает только обезличенные файлы.
