# Эксперименты E007–E025

Все QC-числа ниже — **исторические source-study proxy** на групповых фолдах, а не официальная точность по отдельным DXA-снимкам. Протокол: пять внешних групповых фолдов, семена 17/29/43, inner-only выбор checkpoint, calibration и thresholds; outer predictions только для оценки. Сравнивайте числа лишь при одинаковом протоколе. Подробности и интервалы: [реестр экспериментов](context/EXPERIMENTS.md) и локальные `artifacts/`.

| Эксперимент | Что проверяли | Proxy macro-F1 нарушений | Вывод |
|---|---|---:|---|
| E007 A3 | Замороженные DINO/MI2 + MIL | 0.24697 | Текущий базовый вариант |
| E008 B0 | Упрощённый baseline | 0.0484 | Ниже E007 |
| E010 | Mean+max pooling | 0.2240 | Ниже E007 |
| E011 | Внешний hip spatial adapter | 0.1908 | Нет улучшения нарушений |
| E012 | Более сильный positive weighting | 0.2470 | Практически без улучшения |
| E014 | Частичный pixel stem transfer | 0.1755 | Ниже E007 |
| E015 / E016 | Целый pretrained pixel encoder / scratch контроль | 0.1321 / 0.1708 | Pretraining хуже контроля; парная CI разности ниже нуля |
| E017 / E018 | Frozen pretrained / frozen random pixel encoder | 0.2093 / 0.2345 | Устойчивой пользы pretraining нет |
| E019 | Asymmetric negative focusing | 0.26843 | Парная 95% CI прироста пересекает ноль; AP и quality хуже E007 |

E009 — аналитический outer-label threshold sweep, оптимистичная верхняя граница, в систему не внедрён. E013 обучил внешний keypoint encoder и проверил pipeline, но не доказал улучшения DXA QC. E020 descriptor pilot дал AP delta −0.000253 и ROC-AUC delta −0.002185 относительно E007, поэтому полную grouped QC оценку не запускали. E021 — аудит preprocessing. E023 router показал 1.0 balanced accuracy на внутренних предварительных anatomy assignments, частично связанных с DINO: это круговое development evidence. E024 независимый radiograph check: 0.899 balanced accuracy (95% group-bootstrap CI .870–.925), hip sensitivity .798, 42/208 hip→spine. E025 read-only replay подтвердил общий study-bag вектор у двух hip изображений; ошибка indexing не выявлена. Ни E024, ни E025 не дают DXA image-level QC accuracy.

Для E007/E019 см. [перепроверку E019](E019_RECHECK_REPORT_2026-09-23.md). Для E020 см. [полный E020 отчёт](E020_FINAL_REPORT.md). Подробный E025 содержит сведения по отдельным файлам и хранится только локально; выше приведён его агрегированный вывод.
