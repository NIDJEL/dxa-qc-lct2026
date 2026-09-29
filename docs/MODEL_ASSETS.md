# Модельные активы

Веса не входят в Git и не встраиваются в Docker image. Поместите их в каталог `DXA_ASSETS_DIR` с той же структурой путей:

| Актив | Размер | SHA256 |
|---|---:|---|
| `models/dinov3-vitl16/model.safetensors` | 1,212,559,808 B | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |
| `models/MedImageInsight/mlflow_model_folder/artifacts/checkpoints/vision_model/medimageinsigt-v1.0.0.pt` | 2,464,060,700 B | `5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414` |
| `artifacts/E007_FULL_V1/A3/seed_17/fold_0/checkpoints/best.pt` | 3,498,109 B | `5f66f59b4498fbad56fb01b6f388db1bcc0b639ff71515ff4da1a51c4bc3ad55` |
| `artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json` | 577,545 B | `3aeb17053c0e155a514ea1d9ab76b730fe72286cf3ba144d77b29bc8cb6cf55f` |

DINOv3 следует получать из [официального репозитория Meta](https://github.com/facebookresearch/dinov3) и соблюдать [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), распространяющуюся на веса. MedImageInsight v1 берётся из [каталога Microsoft](https://ai.azure.com/catalog/models/MedImageInsight?source=microsoft); локальная MLflow-папка содержит MIT LICENSE. Используйте именно классическую open-source модель, не MedImageInsight Premium API. Полная локальная папка содержит код, конфигурацию и vision checkpoint; один `.pt` файл недостаточен.

E007 A3 checkpoint и E023 router — собственные конкурсные активы. Их отдельный архив [DXA_QC_runtime_heads_20260929.zip](https://github.com/NIDJEL/dxa-qc-lct2026/releases/download/heads-20260929/DXA_QC_runtime_heads_20260929.zip) опубликован как предварительный GitHub Release, а не в Git. SHA256 архива: `e8035ff2262c902c1e702580452c490f603f9673df072714153a58744580fd37`. Внутри E007 checkpoint оставлены только исходные model weights, calibration и provenance; optimizer/scheduler/scaler удалены. Его новый SHA256 — `65fe56c3e9def2a422623bb488ae6265457b444e6c3b96178670c4f48e2cf08b`, исходный — в таблице. Сверьте `MANIFEST.json` после распаковки. Архив не содержит DICOM, labels и приватные карты соответствия. Локальный CUDA smoke на трёх DICOM подтвердил идентичные выводы с полным checkpoint; время обработки отличается.

Если сторонние веса недоступны или их условия не приняты пользователем, локальный inference не воспроизводится. Контейнер не обращается к сети во время inference.
