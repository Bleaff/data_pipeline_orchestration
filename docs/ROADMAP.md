# 🗺 Роадмап NEUDC

Согласованный план развития проекта, сгруппированный по четырём трекам:
**фундамент** (`track:foundation`), **CV** (`track:cv`), **UI** (`track:ui`) и **LLM** (`track:llm`).

Актуальный статус задач — в [issues](https://github.com/Bleaff/data_pipeline_orchestration/issues)
и на [project board](https://github.com/users/Bleaff/projects/6). Этот файл фиксирует
общую картину и порядок работ; source of truth по деталям — сами issues.

> Stage 1–7 (переход на PUSH/PULL-транспорт, фабрика узлов, схема конфигов,
> сервис-менеджер и базовый набор узлов) выполнены до заведения трекера
> и в issues не отражены.

---

## 🏗 Трек «Фундамент» — надёжность и производительность ядра

Порядок внутри трека важен: линтеры/типизация (Stage 11) идут перед исправлением
двойного потребления (Stage 10), т.к. баг затрагивает батч-инференс и требует
уверенности в типах; наблюдаемость (Stage 12) — база для автоскейла (Stage 13) и всего трека UI.

| Stage | Issue | Приоритет | Суть |
|---|---|---|---|
| 8 | [#10](https://github.com/Bleaff/data_pipeline_orchestration/issues/10) | high | Opt-in `shared_memory` транспорт для изображений вместо pickle+TCP; только одно-консюмерные рёбра, включается через `NEUDC_SHM_IMAGES=1` |
| 9 | [#11](https://github.com/Bleaff/data_pipeline_orchestration/issues/11) | high | Политика ошибок в узлах: retry с экспоненциальным backoff, dead-letter, конфигурируемая политика на узел, счётчики ошибок |
| 10 | [#12](https://github.com/Bleaff/data_pipeline_orchestration/issues/12) | high | **Баг:** процесс-узел потребляет из mailbox двумя конкурирующими циклами (`BaseProcessNode.run()` + поток из `init_runtime()`) — свести к одному пути исполнения |
| 11 | [#13](https://github.com/Bleaff/data_pipeline_orchestration/issues/13) | medium | Включить ruff + mypy (сейчас закомментированы в pre-commit), довести типизацию, CI-матрица по Python и `pytest` в CI |
| 12 | [#14](https://github.com/Bleaff/data_pipeline_orchestration/issues/14) | high | Настоящая наблюдаемость: глубина очередей, throughput/latency/ошибки на узел, единый реестр метрик Prometheus, Grafana-дашборд |
| 13 | [#15](https://github.com/Bleaff/data_pipeline_orchestration/issues/15) | medium | `replicas: N` на тяжёлый узел (PUSH → N PULL-воркеров), опционально автоскейл по глубине очереди |

## 🎯 Трек «CV» — ценность для авторазметки

Главный по актуальности трек: рынок ушёл к foundation-моделям, а проект пока умеет
по сути только YOLO.

| Issue | Приоритет | Суть |
|---|---|---|
| [#16](https://github.com/Bleaff/data_pipeline_orchestration/issues/16) | high | **Ключевая фича.** Zero-shot / open-vocab разметка (GroundingDINO / Grounded-SAM / SAM2): новые классы по текстовому промпту без обучения; новый backend + узел по существующему паттерну |
| [#18](https://github.com/Bleaff/data_pipeline_orchestration/issues/18) | high | Видео: реализовать заявленный `VideoReader` (файл/RTSP), трекер (ByteTrack / BoT-SORT), пропагация лейблов по треку между кадрами |
| [#17](https://github.com/Bleaff/data_pipeline_orchestration/issues/17) | medium | VLM-узел: капшенинг / VQA / атрибуты / open-vocab классификация кадра или кропа (поверх `LLMBackend` из трека LLM) |
| [#19](https://github.com/Bleaff/data_pipeline_orchestration/issues/19) | medium | Active learning++: uncertainty sampling + диверсити по эмбеддингам (coreset / k-center greedy) |
| [#20](https://github.com/Bleaff/data_pipeline_orchestration/issues/20) | medium | Замкнуть петлю разметки: интеграция CVAT / Label Studio — пре-лейблы → правки человека → дообучение → следующая итерация |
| [#21](https://github.com/Bleaff/data_pipeline_orchestration/issues/21) | low | Экспорт датасета в COCO / YOLO / Pascal VOC |

Также в планах: генерация изображений диффузионными моделями (локальные модели или
hosted API) для аугментации датасетов.

## 🖥 Трек «UI» — control-plane и дашборд

Строится поверх метрик из Stage 12.

| Issue | Приоритет | Суть |
|---|---|---|
| [#22](https://github.com/Bleaff/data_pipeline_orchestration/issues/22) | medium | Control-plane API: FastAPI поверх `PipelineServiceManager` — REST (start/stop/status, конфиги, валидация), live-метрики через WebSocket/SSE, превью кадров |
| [#23](https://github.com/Bleaff/data_pipeline_orchestration/issues/23) | medium | Frontend MVP (read-only): граф пайплайна с health/throughput, метрики в реальном времени, лента кадров с боксами |
| [#24](https://github.com/Bleaff/data_pipeline_orchestration/issues/24) | low | Интерактивный frontend: визуальный сборщик конфигов (генерит валидируемый YAML) + UI ревью пре-лейблов |

## 🤖 Трек «LLM» — языковые модели в инструменте и вокруг него

| Issue | Приоритет | Суть |
|---|---|---|
| [#25](https://github.com/Bleaff/data_pipeline_orchestration/issues/25) | medium | `LLMBackend`: провайдер-абстракция — OpenRouter + локальные Ollama / vLLM / llama.cpp через OpenAI-совместимый endpoint; база для VLM-узла и копайлота |
| [#26](https://github.com/Bleaff/data_pipeline_orchestration/issues/26) | low | Копайлот: управление пайплайном/датасетом на естественном языке — agentic-loop поверх control-plane API (делать последним: нужны API с tools и метрики) |

---

## 🔗 Ключевые зависимости между треками

- **Stage 11 (линтеры/типы) → Stage 10 (баг двойного потребления):** фикс высокорисковый, делается после включения проверок.
- **Stage 12 (метрики) → Stage 13 (автоскейл), #22 (API), #23 (дашборд):** наблюдаемость — общий фундамент.
- **#25 (`LLMBackend`) → #17 (VLM-узел), #26 (копайлот).**
- **#22 (control-plane API) → #23/#24 (frontend), #26 (копайлот).**
