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
