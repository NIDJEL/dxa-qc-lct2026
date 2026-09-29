# E020_EXTERNAL_PRETRAINING — итоговый отчёт

Три внешних направления завершены на seeds 17, 29, 43. DINOv3 оставался замороженным; обучались пространственные адаптеры и головы внешних задач. Это исследовательский перенос, не готовый Teacher/Student.

Консервативный расход GPU: **3.155 часа из 12**, включая извлечение признаков и время владения GPU. Подтверждено **36,015 успешных optimizer steps** по Adam-счётчикам.

## Какие данные действительно обучали веса

| Ветвь | Train уникальных пикселей | Validation | Источники |
|---|---:|---:|---|
| spine | 3152 | 844 | dataset2024 |
| hip | 1032 | 275 | pelvis, mtddh2 |
| vindr | 6740 | 1632 | vindr |

Всего **10,924** разных внешних train-пикселей в оптимизации; это не число независимых пациентов. По источникам: dataset2024: 3152, mtddh2: 713, pelvis: 319, vindr: 6740.

AP/LA dataset2024 объединены по исходному номеру; MTDDH — по исходной папке пациента; VinDr — по study_id. Во всех случаях затем объединены точные дубликаты, и лишь после этого назначен split. Независимость всех пациентов между источниками не доказана. Один повреждённый MTDDH PNG исключён и остаётся в preparation.json. VinDr test release не использован. Копии dataset3.7gb/Data/Dataoxy/MTDDH_for_DXA не добавлялись.

## Результат внешних задач

Координаты — нормированная ошибка ожидаемой позиции в letterbox-изображении, не миллиметры. Pelvis/MTDDH Dice — по маскам/полигонам; шестой MTDDH канал является bbox области keypoints. VinDr Dice — соответствие слабым bbox-маскам патологий, не качество анатомической сегментации.

| Ветвь / seed | Шаги | Лучшая эпоха | Validation loss до → после | Средняя вспомогательная метрика до → после | Минут обучения |
|---|---:|---:|---|---|---:|
| spine / 17 | 3465 | 23 | 5.3407 → 2.0995 | 0.1664 → 0.0188 | 2.3 |
| spine / 29 | 3465 | 18 | 5.2851 → 2.1013 | 0.1606 → 0.0189 | 2.6 |
| spine / 43 | 3465 | 19 | 5.3028 → 2.0999 | 0.1590 → 0.0188 | 2.4 |
| hip / 17 | 1155 | 20 | 1.5355 → 0.1487 | 0.1844 → 0.9209 | 0.8 |
| hip / 29 | 1155 | 19 | 1.5235 → 0.1477 | 0.1407 → 0.9215 | 0.8 |
| hip / 43 | 1155 | 22 | 1.4984 → 0.1499 | 0.1359 → 0.9206 | 0.8 |
| vindr / 17 | 7385 | 16 | 1.7504 → 0.1272 | 0.0058 → 0.3914 | 5.1 |
| vindr / 29 | 7385 | 19 | 1.6715 → 0.1279 | 0.0098 → 0.3773 | 4.8 |
| vindr / 43 | 7385 | 16 | 1.6895 → 0.1284 | 0.0104 → 0.3779 | 4.8 |

Каждая ветвь завершила 35 эпох. Сохранены initial/random.pt, best.pt, last.pt с optimizer state, learning_curve.json, complete.json, SHA256 данных/весов/кода. Оптимизировались все параметры компактного Spatial encoder и task head; большой DINO не оптимизировался. Полные поканальные метрики и support доступны в verification.json.

## Перенос в QC: контролируемый внутренний pilot

Контроль — исходные замороженные E007 A3 checkpoints. Fit: folds 2/3/4 (60 studies); inner: fold 1 (20 studies). Outer fold 0 для pilot не вычислялся. Переносились frozen spatial adapter + task head через пространственные дескрипторы. Сравнение с тем же дескриптором от сохранённой случайной инициализации отделяет эффект внешнего обучения от изменения размерности. Единственные новые QC-параметры — фиксированные регуляризованные residual logistic heads; стандартирование только по fit.

Все показатели ниже относятся к **20 внутренним исследованиям** и являются оптимистичной диагностикой: эта часть уже использовалась для выбора A3 и калибровки. Это не внешний/независимый тест и не официальная image-level оценка.

| Компонент | Δ AP к A3 | Δ AP к random | Δ ROC-AUC к A3 | Δ F1 к A3 | Go |
|---|---:|---:|---:|---:|---|
| spine | 0.0004 | 0.0000 | 0.0015 | 0.0000 | False |
| hip | -0.0012 | -0.0003 | -0.0078 | 0.0000 | False |
| vindr | 0.0004 | 0.0000 | 0.0015 | 0.0000 | False |
| combined | -0.0003 | -0.0001 | -0.0022 | 0.0000 | False |

Spine/VinDr строки усредняют три spine-класса; hip — две hip-метки с pooling сторон; combined — все пять. Критерий go был записан до результатов: AP ≥ +0.02 к A3, AP > 0 к random, ROC-AUC не хуже A3, F1 не ниже A3 более чем на .02. Только заранее зафиксированный combined может открыть полный grouped прогон.

### Все официальные QC-типы: A3 → random → pretrained combined

| Тип | Метрика | A3 | Random | Pretrained | Δ к A3 [95% study bootstrap] |
|---|---|---:|---:|---:|---|
| spine_positioning | f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| spine_positioning | sensitivity | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| spine_positioning | specificity | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] |
| spine_positioning | balanced_accuracy | 0.5000 | 0.5000 | 0.5000 | 0.0000 [0.0000, 0.0000] |
| spine_positioning | roc_auc | 0.5417 | 0.5463 | 0.5463 | 0.0046 [0.0000, 0.0263] |
| spine_positioning | average_precision | 0.2210 | 0.2222 | 0.2222 | 0.0012 [0.0000, 0.0074] |
| spine_positioning | pr_auc_trapezoidal | 0.1276 | 0.1282 | 0.1282 | 0.0006 [0.0000, 0.0037] |
| spine_axis | f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| spine_axis | sensitivity | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| spine_axis | specificity | 0.9825 | 1.0000 | 1.0000 | 0.0175 [0.0000, 0.0556] |
| spine_axis | balanced_accuracy | 0.4912 | 0.5000 | 0.5000 | 0.0088 [0.0000, 0.0278] |
| spine_axis | roc_auc | 0.6140 | 0.6140 | 0.6140 | 0.0000 [0.0000, 0.0000] |
| spine_axis | average_precision | 0.1259 | 0.1259 | 0.1259 | 0.0000 [0.0000, 0.0000] |
| spine_axis | pr_auc_trapezoidal | 0.0630 | 0.0630 | 0.0630 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | f1 | 0.8199 | 0.8199 | 0.8199 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | sensitivity | 0.8889 | 0.8889 | 0.8889 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | specificity | 0.8810 | 0.8810 | 0.8810 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | balanced_accuracy | 0.8849 | 0.8849 | 0.8849 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | roc_auc | 0.9246 | 0.9246 | 0.9246 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | average_precision | 0.8599 | 0.8599 | 0.8599 | 0.0000 [0.0000, 0.0000] |
| spine_foreign_object | pr_auc_trapezoidal | 0.8475 | 0.8475 | 0.8475 | 0.0000 [0.0000, 0.0000] |
| hip_positioning_rotation | f1 | 0.5556 | 0.5556 | 0.5556 | 0.0000 [0.0000, 0.0000] |
| hip_positioning_rotation | sensitivity | 0.3889 | 0.3889 | 0.3889 | 0.0000 [0.0000, 0.0000] |
| hip_positioning_rotation | specificity | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] |
| hip_positioning_rotation | balanced_accuracy | 0.6944 | 0.6944 | 0.6944 | 0.0000 [0.0000, 0.0000] |
| hip_positioning_rotation | roc_auc | 0.7733 | 0.7711 | 0.7689 | -0.0044 [-0.0253, 0.0000] |
| hip_positioning_rotation | average_precision | 0.6131 | 0.6124 | 0.6118 | -0.0013 [-0.0059, 0.0000] |
| hip_positioning_rotation | pr_auc_trapezoidal | 0.5955 | 0.5948 | 0.5943 | -0.0012 [-0.0049, 0.0000] |
| hip_roi | f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| hip_roi | sensitivity | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| hip_roi | specificity | 1.0000 | 1.0000 | 1.0000 | 0.0000 [0.0000, 0.0000] |
| hip_roi | balanced_accuracy | 0.5000 | 0.5000 | 0.5000 | 0.0000 [0.0000, 0.0000] |
| hip_roi | roc_auc | 0.7222 | 0.7111 | 0.7111 | -0.0111 [-0.0345, 0.0000] |
| hip_roi | average_precision | 0.1319 | 0.1307 | 0.1307 | -0.0012 [-0.0093, 0.0000] |
| hip_roi | pr_auc_trapezoidal | 0.0660 | 0.0654 | 0.0654 | -0.0006 [-0.0047, 0.0000] |
| quality_pooled | f1 | 0.5994 | 0.6069 | 0.6069 | 0.0075 [-0.0171, 0.0496] |
| quality_pooled | sensitivity | 0.5625 | 0.5833 | 0.5833 | 0.0208 [0.0000, 0.0833] |
| quality_pooled | specificity | 0.8571 | 0.8476 | 0.8476 | -0.0095 [-0.0303, 0.0000] |
| quality_pooled | balanced_accuracy | 0.7098 | 0.7155 | 0.7155 | 0.0057 [-0.0128, 0.0374] |
| quality_pooled | roc_auc | 0.7679 | 0.7673 | 0.7655 | -0.0024 [-0.0115, 0.0057] |
| quality_pooled | average_precision | 0.6132 | 0.6118 | 0.6102 | -0.0030 [-0.0136, 0.0089] |
| quality_pooled | pr_auc_trapezoidal | 0.5954 | 0.5941 | 0.5925 | -0.0029 [-0.0162, 0.0091] |

AP и trapezoidal PR-AUC считаются отдельно. Bootstrap: 2000 выборок целых studies, одинаковых для моделей/seeds/сторон; undefined counts и все seed-specific интервалы сохранены в pilot/paired_bootstrap.json. Качество — независимые официальные таргеты, не OR нарушений.

## Решение о полном протоколе

Комбинированный кандидат **не прошёл заранее записанный go**. Полный 5-fold QC-прогон не запускался. Поэтому изменение outer QC-метрик для E020 **не измерено**, а не равно нулю. Зафиксированный E007 A3 остаётся контролем: исторический пяти-классовый source-study proxy macro-F1 0.24697. Полученные внутренние цифры нельзя сравнивать с ним как с тем же evaluation split.

## Image-level mapping

Проверены 499 DICOM / 252 уникальных изображений / 100 исходных studies. Pixel-based clustering и локальный просмотр всех миниатюр дали кандидаты: 99 spine, 153 hip. Имена файлов и размер изображения не использовались как router. Одна необычная проекция отмечена умеренной уверенностью. Уверенность качественная, не калиброванная вероятность.

В DICOM отсутствуют явные BodyPartExamined, Laterality и ImageLaterality. Сторона бедра остаётся unknown. Ни одна новая image-level QC-метка не объявлена экспертной: все назначения null, исходные study labels доступны отдельным объектом. Source study ID отделён от DICOM UID. Локальная компактная галерея: mapping/optional_review.jpg; полный manifest: mapping/image_level_manifest.json; просмотр: mapping/review.html. Mapping не использован для выбора модели.

## Остальные ресурсы и следующий этап

hip_joints: в каждой диагностической папке ровно 1000 изображений; в локальной выборке видны похожие снимки с различной резкостью/контрастом. Надёжных исходных patient/augmentation групп нет, диагностические labels не применялись. Масштабный SSL не запускался. 01_DXA_DOMAIN: выборка похожа на DXA с нанесёнными ROI, но происхождение и исходные данные не подтверждены; в оптимизации не использовалась. JPG_DICOM: аудит подтверждает DX/CR/RF, есть существенные повторы MTDDH; новый обучающий корпус из копий не создавался.

Следующий этап — обсуждение результата E020 и устранение неоднозначного соответствия image/side/source labels прежде нового дорогого цикла. Не масштабировать неудачный transfer и не начинать массовую псевдоразметку. E021 автоматически не запущен. Платный врач не планируется.

## Проверка и воспроизведение

Проверены hashes feature cache/records/checkpoints, успешные Adam steps, конечность весов, изменение весов относительно random, лучший checkpoint по внешней validation loss, отсутствие групп/точных дубликатов между splits и 24 replay внутренних вероятностей/калибровок. Доказательства: artifacts/E020_EXTERNAL_PRETRAINING/verification.json. Персональные метаданные, картинки, кэши и веса остаются локально и исключены из Git.

```powershell
.venv/Scripts/python.exe scripts/e020_external.py prepare
.venv/Scripts/python.exe scripts/e020_external.py run --branches spine hip vindr --epochs 35
.venv/Scripts/python.exe scripts/e020_mapping.py
.venv/Scripts/python.exe scripts/e020_transfer.py run
.venv/Scripts/python.exe scripts/e020_statistics.py
.venv/Scripts/python.exe scripts/e020_verify.py
.venv/Scripts/python.exe scripts/e020_report.py
.venv/Scripts/python.exe -m pytest tests/test_e020.py -q
```

Воспроизведение в новом output-каталоге; завершённые артефакты не перезаписывать. Конфигурация и ограничения: docs/E020_PREREGISTRATION.md. Скрипты: scripts/e020_*.py. Checkpoints: artifacts/E020_EXTERNAL_PRETRAINING/{spine,hip,vindr}/seed_{17,29,43}/{best,last,random}.pt.
