# Заметки по бенчмарку LightRAG vs HippoRAG

Рабочие наблюдения для финального отчёта. Дата начала: 2026-05-13.

## 1. Конфигурация эксперимента

- Корпус: `data/corpus/` — 102 коротких документа (Wikipedia-абзацы в формате HotpotQA).
- Вопросы: `data/questions.json` — 50 кейсов с `expected_answer`, мультихоп HotpotQA.
- Оценщик: DeepEval, метрики Answer Relevancy / Faithfulness / Contextual Precision / Contextual Recall / Hallucination.
- LLM для метрик: `gpt-4o-mini` через OpenRouter; `max_completion_tokens=2000`; ретраи и обработка `LengthFinishReasonError` пропатчены в `eval/evaluate.py:71-142`.
- Стораджи: HippoRAG — локальные файлы (FAISS + graph + OpenIE cache); LightRAG — Postgres / Neo4j / Qdrant / Redis (см. `docker-compose.yml`).

## 2. Первый прогон — результаты (baseline до фикса)

| Метрика                | LightRAG (mix) | HippoRAG |
|------------------------|---------------:|---------:|
| Answer Relevancy       | 0.578          | 0.710    |
| Contextual Precision   | 0.260          | 0.523    |
| Contextual Recall      | 0.400          | 0.312    |
| Faithfulness           | 0.908          | 0.367    |
| Hallucination (выше=хуже) | 0.040       | 0.940    |

Графики: `data/answers/comparison.png`, `lightrag_metrics.png`, `hipporag_metrics.png`.

## 3. Ключевая методологическая проблема baseline

Системы сдавали в DeepEval **разные виды `retrieval_context`**:

- **LightRAG (`lightrag_rag/query.py:60-62`, исходная версия):**
  `retrieval_context` = ответ `aquery(mode="naive")`, т.е. **синтез LLM**, а не сырые чанки.
- **HippoRAG (`hipporag_rag/query.py:53-54`):**
  `retrieval_context` = `sol.docs[:num_retrieve]` — реальные top-K документов.

Метрики Faithfulness / Hallucination / Contextual Precision / Contextual Recall сравнивают ответ именно с `retrieval_context`. Поэтому 4 из 5 метрик в baseline оценивают **разные вещи**:
- LightRAG получает «высокий Faithfulness / низкую Hallucination» по построению — его «контекст» уже почти равен ответу.
- HippoRAG получает обратное — ответ часто опирается на параметрическую память LLM поверх 5 переданных чанков.

Содержательно из baseline можно интерпретировать только **Contextual Precision** (HippoRAG 0.523 vs LightRAG 0.260) — графовый PPR-retrieval HippoRAG реально лучше отбирает релевантные пассажи.

Дополнительный шум: у HippoRAG ряд значений Faithfulness = `null` (gpt-4o-mini деградировал и упёрся в `max_completion_tokens`). Обрабатывается корректно в `evaluate.py:201-211` (исключается из среднего), но снижает мощность оценки.

## 4. Поведенческое различие систем (видно по сырым ответам)

- **LightRAG** чаще отвечает «I don't have enough information…» (см. `data/answers/lightrag.json`) — консервативное поведение, штрафуется Answer Relevancy.
- **HippoRAG** отвечает уверенно даже при нерелевантных пассажах (например, «Yes — both were American» при контексте об ирландском писателе) — высокий Answer Relevancy, высокая Hallucination.

Contextual Recall у обеих систем низкий (~0.3-0.4) — корпус из 102 абзацев не покрывает все bridge-сущности мультихоп-вопросов.

## 5. Внесённые исправления

Файл: `lightrag_rag/query.py`.

- Дефолтный режим: `mix` → `hybrid` (per request пользователя; naive давал слабые результаты).
- `retrieval_context` теперь — **сырые чанки** через `rag.aquery_data(question, param=...)`:
  тот же retrieval-пайплайн, что и `aquery`, но без LLM-генерации. Возвращает структуру
  `{"status":"success", "data":{"chunks":[{"content":...}, ...]}}` — извлекается `content`.
- Новый флаг `--num-retrieve` (default=5) — режет экспорт до того же числа, что у HippoRAG, чтобы DeepEval видел сопоставимые входы.
- `chunk_top_k = max(top_k // 2, num_retrieve)` — гарантирует, что нужное число чанков долетает до выхода.

HippoRAG-сторона не менялась: она уже отдаёт сырые `sol.docs`.

`eval/evaluate.py` не менялся: читает `retrieval_context` как есть.

## 6. Команды для повторного прогона

```bash
docker-compose up -d   # поднять Postgres/Neo4j/Qdrant/Redis
uv run scripts/query_lightrag.py --questions data/questions.json \
    --output data/answers/lightrag.json --mode hybrid --num-retrieve 5
uv run scripts/evaluate.py --answers data/answers/lightrag.json \
    data/answers/hipporag.json --compare --charts
```

## 7. Ожидания от честного сравнения (гипотезы, проверить после прогона)

- **Faithfulness LightRAG** упадёт с 0.91 (артефакт уходит) — реальный baseline.
- **Hallucination LightRAG** вырастет с 0.04, но должна остаться ниже HippoRAG за счёт паттерна «отказа отвечать вне контекста».
- **Contextual Precision/Recall** впервые становятся сопоставимыми; ожидаю сокращение разрыва, возможно LightRAG обгонит за счёт `hybrid` (entities + relations + chunks).
- **Answer Relevancy** не зависит от `retrieval_context` — почти не изменится; небольшой сдвиг возможен из-за `mix → hybrid`.

## 8. TODO для финального отчёта

- [ ] Прогнать новый `lightrag.json` после фикса; сохранить как baseline_v2.
- [ ] Сохранить старые eval-файлы под суффиксом `_v1` перед перезаписью (для сравнения «до/после фикса»).
- [ ] Сравнить, изменился ли Contextual Recall — это единственная метрика, которая не должна сильно зависеть от типа контекста.
- [ ] Проверить, остались ли `null` у Faithfulness HippoRAG после ретрая — если да, рассмотреть смену модели оценщика.
- [ ] Расширить корпус или ограничить вопросы теми, для которых bridge-сущности есть в `data/corpus`, чтобы поднять Contextual Recall и сделать выводы значимыми.
- [ ] Добавить в отчёт раздел про стоимость/латентность: HippoRAG — локальные эмбеддинги bge-m3 (dim=1024), LightRAG — OpenAI embeddings (dim=1536) + 4 внешних БД.

## 9. Второй прогон после фикса — результаты (2026-05-14)

Файлы: `data/answers/lightrag_eval.json` (n=20), `data/answers/hipporag_eval.json` (n=50), графики `comparison.png`, `lightrag_metrics.png`, `hipporag_metrics.png`.

| Метрика                | LightRAG (hybrid, n=20) | HippoRAG (n=50) | Δ к baseline (LightRAG) |
|------------------------|------------------------:|----------------:|------------------------:|
| Answer Relevancy       | 0.545                   | 0.700           | 0.578 → 0.545 (≈)       |
| Faithfulness           | 0.868                   | 0.367           | 0.908 → 0.868 (слабо ↓) |
| Hallucination (хуже=↑) | 0.880                   | 0.936           | 0.040 → 0.880 (↑↑)      |
| Contextual Precision   | 0.364                   | 0.507           | 0.260 → 0.364 (↑)       |
| Contextual Recall      | 0.270                   | 0.292           | 0.400 → 0.270 (↓)       |

### 9.1 Сверка с гипотезами §7

- ✅ **Hallucination LightRAG ↑ с 0.04** — артефакт «контекст=ответ» снят.
- ✅ **Hallucination LightRAG остаётся ниже HippoRAG** — но маржа маленькая (0.88 vs 0.94).
- ✅ **Answer Relevancy почти не изменилась** (0.578 → 0.545).
- ❌ **Contextual Precision LightRAG не обогнал HippoRAG** — HippoRAG всё ещё впереди (0.507 vs 0.364). Графовый PPR HippoRAG сильнее `hybrid` LightRAG.
- ❌ **Faithfulness LightRAG почти не упал** (0.91 → 0.87) — неожиданно. Либо `aquery_data` всё ещё отдаёт частично пост-обработанный текст, либо `hybrid`-контекст (entity+relation+chunk) реально хорошо удерживает генерацию. Требует ручной проверки сырых записей в `lightrag.json`.

### 9.2 Содержательные выводы (с оговорками)

- HippoRAG **лучше отбирает чанки** (Contextual Precision +0.14) — графовый PPR-retrieval работает.
- HippoRAG **отвечает увереннее, но галлюцинирует** (AR 0.70, Hallucination 0.94) — опирается на параметрическую память LLM поверх top-K.
- LightRAG в `hybrid` **сохраняет высокую Faithfulness** (0.87) даже на сырых чанках — это уже не артефакт baseline.
- **Hallucination 0.88–0.94 у обеих систем** — обе почти всегда галлюцинируют; метрика близка к потолку и плохо различает.
- **Contextual Recall 0.27–0.29 у обеих** — корпус не покрывает bridge-сущности; метрика шумная (см. TODO §8 про расширение корпуса).

### 9.3 Методологические проблемы — выводы НЕ однозначны

1. **Несопоставимые `n`: LightRAG = 20, HippoRAG = 50.** Прогон LightRAG обрывается на 20-м вопросе (`lightrag_eval.json` содержит ровно 20 кейсов, hipporag_eval.json — 50). Сравнение средних некорректно. При срезе HippoRAG по первым 20 вопросам Answer Relevancy ≈ 0.65 — часть разрыва 0.70 vs 0.55 схлопывается.
2. **Малая выборка (20–50)** — широкие доверительные интервалы, разница ≤ 0.10 на бинарных шкалах может быть шумом.
3. **`null` у Faithfulness HippoRAG** всё ещё встречаются (gpt-4o-mini → `max_completion_tokens`). По коду исключаются из среднего, но снижают мощность для главной метрики «верности контексту». См. TODO §8.
4. **Hallucination на потолке** (~0.9 у обеих) — мало информации, нужна более чувствительная метрика или фильтрация вопросов без покрытия в корпусе.
5. **Faithfulness LightRAG подозрительно высокая** — нужно глазами проверить `retrieval_context` в `lightrag.json` и убедиться, что это сырые `content` чанков, а не результат `aquery`.

### 9.4 Дополнительные TODO к §8

- [ ] **Доделать прогон LightRAG на полных 50 вопросах** (сейчас обрывается на 20 — главная блокирующая проблема для статистически валидного сравнения).
- [ ] Спот-чек 3–5 записей `lightrag.json`: убедиться, что `retrieval_context[i]` — сырой текст чанка, не synth-ответ. Подтвердить корректность фикса §5.
- [ ] Посчитать HippoRAG-метрики по тем же 20 вопросам, что у LightRAG, для прямого сравнения до полного прогона.
- [ ] `hipporag_4*.json` — это smoke-тест на 4 кейсах, для финального сравнения не используется; пометить в репо или удалить.
