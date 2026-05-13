# Graph RAG Benchmark: LightRAG vs HippoRAG 2

Сравнение двух граф-базированных RAG-фреймворков на единых метриках DeepEval.

## Архитектурные различия

| | **LightRAG** | **HippoRAG 2** |
|---|---|---|
| Граф | Knowledge Graph (сущности + связи) | Knowledge Graph из OpenIE-триплетов |
| Retrieval | Dual-level: векторный + граф | PPR (Personalized PageRank) по KG |
| Внешние БД | PostgreSQL, Neo4j, Qdrant, Redis | только локальные файлы (.parquet, FAISS) |
| Multi-hop | Частично (через граф) | Специализирован на multi-hop QA |
| Обновление | Инкрементальное | Инкрементальное (кэш по хешу) |

## Стек баз данных (LightRAG)

PostgreSQL  -> KV storage    (чанки, метаданные сущностей)
Neo4j       -> Graph storage  (граф знаний: узлы + рёбра)
Qdrant      -> Vector storage (эмбеддинги сущностей, чанков, связей)
Redis       -> Doc status     (состояние пайплайна индексирования)

## Быстрый старт

1. Настройка окружения:
   cp .env.example .env
   # Укажи OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL

2. Запуск баз данных:
   docker compose up -d

3. Подготовка датасета:
   uv run scripts/prepare_dataset.py --dataset hotpotqa --limit 200
   # Ограничить и вопросы, и корпус (для экономии кредитов LLM):
   uv run scripts/prepare_dataset.py --dataset hotpotqa --limit 20 --corpus-limit 50
   # ИЛИ из Obsidian vault:
   uv run scripts/prepare_dataset.py --obsidian /path/to/vault

4. Индексирование:
   uv run scripts/ingest_lightrag.py --docs-dir data/corpus --batch-size 5
   uv run scripts/ingest_hipporag.py --docs-dir data/corpus
   # Ограничить кол-во документов (если корпус уже подготовлен):
   uv run scripts/ingest_lightrag.py --docs-dir data/corpus --max-docs 50
   uv run scripts/ingest_hipporag.py --docs-dir data/corpus --max-docs 50

5. Запросы:
   uv run scripts/query_lightrag.py --questions data/questions.json --output data/answers/lightrag.json
   uv run scripts/query_hipporag.py --questions data/questions.json --output data/answers/hipporag.json

6. Оценка и сравнение:
   uv run scripts/evaluate.py \
       --answers data/answers/lightrag.json data/answers/hipporag.json \
       --compare --charts

## Визуализация графа знаний

### HippoRAG — интерактивный HTML (pyvis)

После индексирования запусти:

    uv run scripts/visualize_hipporag.py
    # → data/graph_hipporag.html  (открой в браузере)

Опции:

    --out data/my_graph.html      # путь к выходному HTML
    --max-nodes 500               # лимит узлов (default: 1000)
    --no-chunks                   # только сущности, без пассажей
    --graphml data/graph.graphml  # дополнительно экспорт GraphML

Легенда: синие точки — сущности (entity), красные квадраты — пассажи (chunk).

### Экспорт в Gephi / yEd

    uv run scripts/visualize_hipporag.py --graphml data/graph.graphml
    # Открой data/graph.graphml в Gephi или yEd

### Визуализация корпуса через /graphify (Claude Code)

Для документоориентированного графа (кластеры тем, связи между файлами):

    # В Claude Code CLI:
    /graphify data/corpus

Генерирует интерактивный HTML + GraphRAG-ready JSON + GRAPH_REPORT.md
прямо из исходных документов (независимо от HippoRAG).

## Ограничение объёма данных (экономия кредитов LLM)

Оба фреймворка вызывают LLM при каждом документе во время индексирования.
Чтобы снизить расходы, используй следующие флаги:

| Флаг | Скрипт | Эффект |
|---|---|---|
| `--corpus-limit N` | `prepare_dataset.py` | Сохраняет не более N документов в `data/corpus/` |
| `--limit N` | `prepare_dataset.py` | Сохраняет не более N QA-пар в `data/questions.json` |
| `--max-docs N` | `ingest_lightrag.py` | Обрабатывает первые N файлов из `data/corpus/` |
| `--max-docs N` | `ingest_hipporag.py` | Обрабатывает первые N файлов из `data/corpus/` |

Рекомендуемые настройки для smoke-теста (~50 LLM-вызовов):

    uv run scripts/prepare_dataset.py --dataset hotpotqa --limit 20 --corpus-limit 50
    uv run scripts/ingest_lightrag.py --docs-dir data/corpus --batch-size 5 --resume
    uv run scripts/ingest_hipporag.py --docs-dir data/corpus

Флаг `--resume` у LightRAG позволяет продолжить с места остановки при повторном запуске.

## Различие эмбеддинг-моделей (важно для отчёта)

HippoRAG и LightRAG используют **разные** эмбеддинг-модели с разными размерностями:

| Фреймворк | Модель | Размерность | Источник |
|---|---|---|---|
| **LightRAG** | `text-embedding-3-small` | **1536** | OpenAI API (через OpenRouter) |
| **HippoRAG** | `BAAI/bge-m3` | **1024** | Локально (sentence-transformers) |

**Причина расхождения:** OpenRouter не поддерживает `/v1/embeddings` endpoint для сторонних
моделей — HippoRAG обращался к `api.openai.com` напрямую с OpenRouter-ключом и получал 401.
Решение: HippoRAG переведён на локальную модель BAAI/bge-m3 через `sentence-transformers`.

**Следствие для метрик:** оба фреймворка строят свои индексы в независимых векторных
пространствах (1024 vs 1536), что является **неустранимым ограничением** при сравнении
косинусных сходств между фреймворками. Однако для внутреннего retrieval каждого фреймворка
это не влияет на корректность — важно зафиксировать в отчёте.

**Конфигурационные переменные** (`.env`):
```
EMBEDDING_MODEL=text-embedding-3-small   # LightRAG
EMBEDDING_DIM=1536
HIPPORAG_EMBEDDING_MODEL=Transformers/BAAI/bge-m3  # HippoRAG
HIPPORAG_EMBEDDING_DIM=1024
```

## Метрики DeepEval

- Answer Relevancy     : отвечает ли ответ на вопрос?
- Faithfulness         : основан ли ответ на найденном контексте?
- Hallucination        : есть ли факты, отсутствующие в контексте?
- Contextual Precision : релевантен ли найденный контекст (без шума)?
- Contextual Recall    : содержит ли контекст нужные факты? (требует expected_answer)

## Рекомендуемые датасеты

HotpotQA (distractor) -- multi-hop QA, требует связи фактов из разных документов
MuSiQue              -- multi-hop (2-4 хопа), сложнее HotpotQA
2WikiMultiHopQA      -- кросс-документное рассуждение
SQuAD v2             -- single-hop, базовый санити-чек
RGB                  -- тест на noise robustness специфично для RAG

## Obsidian vault как датасет

Vault ~/Projects/ProjectsRootVault содержит 144 заметки (~2.8 MB)
по темам RAG, RecSys, TinyML, prompt engineering.

Подходит для smoke-тестирования — структурированный контент по конкретной области.
Ограничения: нет gold QA пар (Contextual Recall недоступен), небольшой объём.
Рекомендация: vault для качественного анализа, HotpotQA/MuSiQue для количественного.
