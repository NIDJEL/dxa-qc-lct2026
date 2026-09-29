# DXA QC Web — локальный веб-сервис

Локальный интерфейс принимает DICOM-файлы, запускает существующий frozen E007 A3 + DINOv3 Large + MedImageInsight + E023 anatomy-router и показывает реальные результаты inference. Медицинские изображения не отправляются во внешние API.

## Запуск в Windows

Из корня проекта:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_web.ps1
```

Скрипт проверяет `.venv`, запускает один backend-процесс и открывает `http://127.0.0.1:8000/`. Если порт занят:

```powershell
$env:DXA_QC_PORT = "8001"
.\run_web.ps1
```

Для ручного запуска: `\.venv\Scripts\python.exe scripts\run_dxa_qc_web.py`. Модели загружаются один раз на процесс и переиспользуются между заданиями. Одновременно выполняется только одно тяжёлое inference-задание; ещё максимум два задания ожидают в очереди.

## Сценарий пользователя

1. Перетащите один DICOM, несколько файлов или папку.
2. Нажмите **Начать анализ** и дождитесь результатов.
3. Используйте фильтры и откройте карточку снимка для подробностей.
4. Скачайте JSON-аудит. CSV доступен только для многоснимочного режима, но ещё требует подтверждения шаблона организаторов.

Границы исследований определяются по DICOM `StudyInstanceUID`/`SeriesInstanceUID`. Разные исследования в одном задании не объединяются в один MIL-bag. Повреждённые файлы остаются в JSON как ошибки, независимые корректные группы продолжают обработку.

## Что означают результаты

- **Позвоночник / бедро** — маршрут замороженного E023 router.
- **Качество** — отдельная quality-head E007, не логическое OR нарушений.
- **Нарушения** — вероятности применимых голов и сравнение с сохранённым checkpoint threshold.
- Вероятность router не является клинически откалиброванной уверенностью.
- E007 targets — study-level proxy, повторяемый у изображений одного bag; это не подтверждённая image-level точность и не диагноз.
- Для бедра сторона не подтверждена; правые/левые головы объединяются максимумом для review.
- Геометрические контуры, heatmap, landmarks и bounding boxes не рисуются: валидированной геометрической модели нет.

## Экспорт

**Скачать подробный JSON** содержит строки изображений, все десять исходных вероятностей голов, thresholds, temperatures, checkpoint/router provenance, grouping, geometry, ошибки и предупреждения.

**Скачать CSV** содержит только восемь полей:

```text
path_to_study,study_uid,image_uid,anatomical_region,quality_class,violation_type,processing_status,time_of_processing
```

Окончательная совместимость CSV, правила `Failure` и `unknown anatomy` ещё не подтверждены организаторами. Ошибки сохраняются в JSON; CSV ограничен успешными строками до получения политики организаторов. Заголовок скачивания содержит предупреждение. Не следует считать CSV финальным конкурсным контрактом.

## Singleton-bag

В форме есть экспериментальный режим «каждый снимок отдельно». Он нужен для read-only сравнения с корректным study-bag. Его результаты предварительные: точность не подтверждена, и официальный CSV отключён. Для одиночного файла режим также помечается как предварительный, поскольку один снимок не заменяет подтверждённую study-bag обработку.

## Docker

```bash
docker build -t dxa-qc:web .
docker run --rm --gpus all --network none -p 8000:8000 \
  -e DXA_QC_DEVICE=cuda \
  -v /absolute/path/models:/app/models:ro \
  -v /absolute/path/artifacts/E007_FULL_V1/A3:/app/artifacts/E007_FULL_V1/A3:ro \
  -v /absolute/path/artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json:/app/artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json:ro \
  dxa-qc:web
```

Существующий CLI не удалён: `scripts/run_dxa_qc_service.py`; `run_container.sh` явно переопределяет entrypoint web-образа на CLI.

## Ограничения и приватность

Поддерживается существующий audited формат: single-frame, 2-D, `uint8`, `MONOCHROME2` DICOM. Веса, калибровка и thresholds не менялись. Временные uploads хранятся локально в `tmp/dxa_qc_web_jobs` и удаляются после TTL или через DELETE API. Размер задания ограничен 512 файлами и 512 МБ; один файл — 128 МБ. Не храните медицинские данные в публичных каталогах.

Официальная отправка остаётся **BLOCKED**: нет подтверждённого evaluator/шаблона для mixed-study, Failure/unknown и независимой expert image-level DXA validation.

Read-only пример сравнения запускается отдельно:

```powershell
.\.venv\Scripts\python.exe scripts\compare_singleton_bags_read_only.py `
  "Dataset_learn\Исследования\<known-study>" `
  --output artifacts\singleton_vs_study_bag_read_only_20260924.json `
  --device cuda
```

Локальный пример: 6 файлов/3 уникальных изображения, максимальная разница
вероятности 0.823 и 11/30 различий threshold-решений. Это подтверждает
предварительный статус singleton, но не является оценкой точности.
