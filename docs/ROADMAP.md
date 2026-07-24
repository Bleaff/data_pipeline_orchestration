# 🗺 Роадмап NEUDC

Согласованный план развития проекта, сгруппированный по пяти трекам:
**фундамент** (`track:foundation`), **CV** (`track:cv`), **UI** (`track:ui`),
**LLM** (`track:llm`) и **Realtime** (`track:realtime`).

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
| — | [#39](https://github.com/Bleaff/data_pipeline_orchestration/issues/39) | medium | **Баг:** health-монитор помечает штатно простаивающий узел как unhealthy — живость цикла путается с наличием входящих данных |
| — | [#40](https://github.com/Bleaff/data_pipeline_orchestration/issues/40) | medium | **Баг:** `if result:` вместо `if result is not None:` — латентный, выстрелит на новых payload-типах (#34). Чинить **до** их мержа |

### Эпик «Data streams» — от кадров к произвольным потокам

Граф узлов пока жёстко завязан на `Frame`. Транспорт уже payload-агностичен
(#10 / PR #32: любой крупный out-of-band буфер едет через shared memory), так что
эпик — про типы сообщений и контракты узлов, а не про передачу.

| Issue | Приоритет | Суть |
|---|---|---|
| [#33](https://github.com/Bleaff/data_pipeline_orchestration/issues/33) | medium | Базовый тип `BaseMessage`, от которого наследуется `Frame`; `Batch` — дженерик над ним; mailbox типизируется по нему вместо `Batch \| Frame` |
| [#34](https://github.com/Bleaff/data_pipeline_orchestration/issues/34) | low | Payload-схемы: `TextChunk`, `TokenTensor`, `AudioChunk`, `VideoSegment` |
| [#35](https://github.com/Bleaff/data_pipeline_orchestration/issues/35) | low | Контракт узла: явная декларация `accepts` / `emits`, диспетч в фабрике |
| [#36](https://github.com/Bleaff/data_pipeline_orchestration/issues/36) | low | Config-валидатор: совместимость payload-типов на рёбрах, fail-fast с именем узла |

Порядок строгий: **#33 → #34 → #35 → #36.**

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

## 🎙 Трек «Realtime» — потоковый инференс

Пайп проектировался под офлайн-разметку: пропускная способность важнее задержки,
терять данные нельзя. Диалоговые сценарии (mic → VAD → ASR → LLM → TTS) переворачивают
оба допущения. Эпик «Data streams» закрывает измерение **«тип»** — какой payload едет
по графу; этот трек закрывает измерение **«время»**. Без второго получится пайп,
который умеет носить аудио и при этом тормозит.

| Issue | Приоритет | Суть |
|---|---|---|
| [#37](https://github.com/Bleaff/data_pipeline_orchestration/issues/37) | high | `process()` как генератор: N сообщений на один вход, инкрементально. Сейчас контракт строго 1-in → 1-out, а `Batch` требует накопить весь результат целиком — для потоковых ASR / LLM / TTS это задержка в размер всей реплики |
| [#38](https://github.com/Bleaff/data_pipeline_orchestration/issues/38) | high | Политика очереди на ребро: `block` (дефолт, как сейчас) / `drop_oldest` / `conflate`. Разметке нужен lossless, реалтайму — выкинуть протухшее и не копить лаг |
| [#41](https://github.com/Bleaff/data_pipeline_orchestration/issues/41) | medium | Приоритетный control-канал (barge-in): обратное ребро собирается уже сейчас, но команда «замолчи» встаёт в тот же FIFO за бэклогом — нужен путь мимо очереди данных |
| [#42](https://github.com/Bleaff/data_pipeline_orchestration/issues/42) | low | Живой аудио-источник (микрофон / поток) + VAD. Та же задача, что `VideoReader` в #18: источник задаёт темп, не имеет конца, может отставать |

Порядок: **#37 первым** — самая рискованная архитектурно правка, цену надо выяснить
до того, как поверх наросли узлы.

---

## 🔗 Ключевые зависимости между треками

- **Stage 11 (линтеры/типы) → Stage 10 (баг двойного потребления):** фикс высокорисковый, делается после включения проверок.
- **Stage 12 (метрики) → Stage 13 (автоскейл), #22 (API), #23 (дашборд):** наблюдаемость — общий фундамент.
- **#25 (`LLMBackend`) → #17 (VLM-узел), #26 (копайлот).**
- **#22 (control-plane API) → #23/#24 (frontend), #26 (копайлот).**
- **#33 (`BaseMessage`) → весь эпик Data streams, а также #41:** `session_id` / `turn_id` /
  `is_final` живут в базовом типе, иначе их придётся вносить вторым проходом по всем узлам.
- **#40 (truthiness) → #34 (payload-схемы):** починить *до*, иначе пустой `AudioChunk`
  или `TextChunk("")` будет молча теряться без следа в логах и метриках.
- **#37 (генератор) → #41 (отмена реплики):** прерывать нечего, пока узел отдаёт
  ровно одно сообщение на вход.
- **#34 (`AudioChunk`) + #38 (политика очереди) → #42 (аудио-источник).**
- **#38 (счётчик дропов) ↔ #14 (метрики):** дроп без счётчика — потеря без следа.
