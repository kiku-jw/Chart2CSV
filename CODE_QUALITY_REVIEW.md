# Отчет о качестве кода и документации Chart2CSV

**Дата проверки:** 2025-12-21
**Проверено:** Claude Code (Sonnet 4.5)
**Ветка:** `claude/review-code-quality-X326M`

---

## Резюме

**Общая оценка:** ⭐⭐⭐⭐ (4/5)

Проект демонстрирует хорошую архитектуру и организацию кода, но имеет несколько критических проблем, требующих внимания.

### Сильные стороны ✅

- Хорошо структурированная модульная архитектура
- Комплексная система типов с использованием dataclasses
- Детальная система отслеживания уверенности и предупреждений
- Продакшн-готовое API с FastAPI
- Двухэтапная стратегия LLM-экстракции

### Критические проблемы 🚨

1. **Безопасность:** Небезопасная обработка путей и отсутствие валидации
2. **Тестирование:** Минимальное покрытие тестами (~5%)
3. **Документация:** Неполная информация в setup.py
4. **Конфигурация:** Отсутствуют файлы линтинга (pyproject.toml, .flake8)
5. **Обработка ошибок:** Неконсистентная в некоторых местах

---

## 1. Безопасность (Security) 🔐

### 🚨 Критические проблемы

#### 1.1 Небезопасная обработка путей в API

**Файл:** `api/main.py:27-28`

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Проблема:** Манипуляция `sys.path` в продакшн коде - плохая практика. Это может привести к импорту неожиданных модулей.

**Рекомендация:** Использовать правильную установку пакета через `pip install -e .`

---

#### 1.2 Отсутствие валидации файлов

**Файл:** `api/main.py:126-135`

```python
def image_to_temp_path(image_bytes: bytes) -> str:
    import tempfile
    img = Image.open(io.BytesIO(image_bytes))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        return f.name
```

**Проблемы:**
- Нет проверки формата изображения (может быть zip-бомба)
- Нет ограничения на размер распакованного изображения
- `delete=False` создает временные файлы, которые могут не удалиться при сбое

**Рекомендация:**
```python
# Добавить проверку размеров
MAX_IMAGE_PIXELS = 89478485  # PIL default
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Проверить формат
if img.format not in ['PNG', 'JPEG', 'WEBP']:
    raise ValueError(f"Unsupported format: {img.format}")

# Проверить размеры
if img.width * img.height > MAX_IMAGE_PIXELS:
    raise ValueError("Image too large")
```

---

#### 1.3 Утечка секретов

**Файл:** `chart2csv/core/llm_extraction.py:50-52`

```python
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not set")
```

**Проблема:** Ошибка может логироваться с контекстом, раскрывая факт использования API ключа.

**Рекомендация:** Использовать более безопасное сообщение об ошибке.

---

#### 1.4 CORS открыт для всех

**Файл:** `api/main.py:108-114`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🚨 Небезопасно!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Проблема:** `allow_origins=["*"]` с `allow_credentials=True` - серьезная уязвимость безопасности.

**Рекомендация:**
```python
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["https://kikuai-lab.github.io"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)
```

---

### ⚠️ Предупреждения безопасности

#### 1.5 Отсутствие аутентификации

**Файл:** `api/main.py`

Rate limiting есть (20 req/min), но нет:
- API ключей для аутентификации
- Логирования запросов
- Защиты от CSRF

**Рекомендация:** Добавить API keys или JWT токены для продакшн использования.

---

## 2. Качество кода (Code Quality) 💻

### ⚠️ Проблемы качества

#### 2.1 Дублирование кода в API endpoints

**Файл:** `api/main.py:183-335, 338-419, 422-515`

**Проблема:** Три endpoint'а (`/extract`, `/extract/base64`, `/extract/calibrated`) содержат дублирующуюся логику:
- Проверка rate limit (3 раза)
- Сохранение в temp файл (3 раза)
- Построение CSV (3 раза)
- Cleanup temp файлов (3 раза)

**Рекомендация:** Выделить общую логику в helper функции:

```python
async def _process_extraction(
    image_bytes: bytes,
    mode: str,
    chart_type: Optional[str],
    calibration: Optional[dict] = None
) -> ExtractionResult:
    # Общая логика обработки
    pass
```

---

#### 2.2 Магические числа

**Файл:** `chart2csv/core/types.py:85-90`

```python
def overall(self) -> float:
    return (
        0.3 * self.crop +      # Откуда эти веса?
        0.25 * self.axis +
        0.3 * self.ocr +
        0.15 * self.extraction
    )
```

**Проблема:** Магические числа без объяснения.

**Рекомендация:**
```python
# Константы с документацией
CONFIDENCE_WEIGHTS = {
    'crop': 0.3,      # Crop is critical - affects all downstream
    'axis': 0.25,     # Axis detection enables transformation
    'ocr': 0.3,       # OCR determines scale accuracy
    'extraction': 0.15  # Extraction confidence is relative
}
```

---

#### 2.3 Неконсистентная обработка ошибок

**Файл:** `chart2csv/core/llm_extraction.py:154-157`

```python
except json.JSONDecodeError as e:
    return {"error": f"JSON parse error: {e}"}, 0.0
except Exception as e:
    return {"error": str(e)}, 0.0
```

**Проблема:** Слишком широкий `except Exception`. Скрывает реальные ошибки.

**Рекомендация:** Перехватывать конкретные исключения.

---

#### 2.4 Потенциальная ошибка в pipeline.py

**Файл:** `chart2csv/core/pipeline.py:158-163`

```python
transform, fit_error = build_transform(
    ticks=ticks,
    x_scale=x_scale,
    y_scale=y_scale
)
if fit_error > 0.1 and x_scale == Scale.LINEAR and y_scale == Scale.LINEAR:
```

**Проблема:** Переменная `fit_error` используется, но не определена в случае `calibration_points` (строка 152-156).

**Потенциальный баг:** UnboundLocalError при определенных условиях.

---

#### 2.5 Устаревший модуль mock

**Файл:** `chart2csv/tests/test_mistral.py:4`

```python
from unittest.mock import MagicMock, patch
```

Правильно! Но в Python 3.3+ это стандарт. Никаких проблем.

---

### ✅ Хорошие практики

1. **Type hints везде** - отличная типизация с использованием dataclasses
2. **Enum для констант** - `ChartType`, `Scale`, `WarningCode`
3. **Dataclasses** - чистые структуры данных
4. **Docstrings** - присутствуют в большинстве функций
5. **Модульная архитектура** - четкое разделение ответственности

---

## 3. Документация (Documentation) 📚

### ✅ Сильные стороны

1. **README.md** - хорошо структурирован:
   - Quick start
   - Структура проекта
   - API endpoints
   - Установка

2. **Модульные README** - в `/api`, `/chart2csv`, `/scripts`, `/deploy`

3. **Docstrings** - детальные в core модулях:
   - `pipeline.py:45-47` - параметры функций
   - `types.py:74-84` - формула confidence с документацией
   - `llm_extraction.py:30-49` - полное описание API

4. **Комментарии в коде** - объясняют сложные алгоритмы

---

### ⚠️ Недостатки документации

#### 3.1 setup.py содержит placeholder данные

**Файл:** `setup.py:26-31`

```python
author="Your Name",
author_email="your.email@example.com",
url="https://github.com/yourusername/chart2csv",
```

**Проблема:** Неактуальные данные автора и URL.

**Рекомендация:**
```python
author="KikuAI",
author_email="contact@kikuai.dev",
url="https://github.com/KikuAI-Lab/Chart2CSV",
```

---

#### 3.2 Устаревшая лицензия в setup.py

**Файл:** `setup.py:39`

```python
"License :: OSI Approved :: MIT License",
```

**Проблема:** Проект использует AGPL-3.0 (см. `README.md:11`, коммит `70f92cd`), но в setup.py указан MIT.

**Рекомендация:**
```python
"License :: OSI Approved :: GNU Affero General Public License v3",
```

---

#### 3.3 Отсутствует CONTRIBUTING.md

Нет гайда для контрибьюторов:
- Как запустить тесты
- Как настроить dev окружение
- Code style guide

---

#### 3.4 Отсутствует CHANGELOG.md

Нет истории изменений версий.

---

#### 3.5 Неполная документация API

**Файл:** `api/main.py`

API endpoints имеют docstrings, но:
- Нет примеров curl команд с calibration_json
- Нет описания формата ответа при ошибках
- Нет информации о retry политике

---

#### 3.6 Отсутствуют примеры использования

Нет директории `/examples` с:
- Примерами Python скриптов
- Примерами изображений
- Jupyter notebook tutorial

---

## 4. Тестирование (Testing) 🧪

### 🚨 Критическая проблема: Низкое покрытие

**Текущее покрытие:** ~5% (1 тестовый файл, 88 строк кода)

**Покрыто тестами:**
- ✅ `chart2csv/core/mistral_ocr.py` - частично
- ✅ `chart2csv/core/ocr.py` - частично (через mocks)

**НЕ покрыто тестами:**
- ❌ `api/main.py` (521 строка) - 0%
- ❌ `chart2csv/core/pipeline.py` (279 строк) - 0%
- ❌ `chart2csv/core/extraction.py` - 0%
- ❌ `chart2csv/core/detection.py` - 0%
- ❌ `chart2csv/core/transform.py` - 0%
- ❌ `chart2csv/core/llm_extraction.py` - 0%
- ❌ CLI модули - 0%

---

### Рекомендации по тестированию

1. **Unit тесты:**
   ```python
   tests/
   ├── test_pipeline.py
   ├── test_detection.py
   ├── test_transform.py
   ├── test_extraction.py
   └── test_types.py
   ```

2. **Integration тесты:**
   ```python
   tests/integration/
   └── test_api.py  # FastAPI TestClient
   ```

3. **Fixture данные:**
   ```python
   tests/fixtures/
   ├── sample_line_chart.png
   ├── sample_scatter.png
   └── expected_results.json
   ```

4. **CI/CD:**
   ```yaml
   # .github/workflows/test.yml
   - pytest --cov=chart2csv --cov-report=xml
   - coverage report --fail-under=70
   ```

---

## 5. Конфигурация проекта (Configuration) ⚙️

### ❌ Отсутствующие файлы

#### 5.1 pyproject.toml

Нет современного `pyproject.toml`. Рекомендуется заменить `setup.py`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chart2csv"
version = "0.1.0"
authors = [{name = "KikuAI", email = "contact@kikuai.dev"}]
license = {text = "AGPL-3.0"}
requires-python = ">=3.8"

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.mypy]
python_version = "3.8"
strict = true
warn_return_any = true

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["chart2csv/tests"]
python_files = "test_*.py"
```

---

#### 5.2 .gitignore

Проверим наличие:

```bash
# Должно быть в .gitignore:
__pycache__/
*.py[cod]
.env
.pytest_cache/
.mypy_cache/
*.egg-info/
dist/
build/
temp_*.png
```

---

#### 5.3 pre-commit hooks

Отсутствуют. Рекомендация: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.280
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
```

---

## 6. Архитектура (Architecture) 🏗️

### ✅ Отличные решения

1. **Модульный pipeline:**
   ```
   Preprocess → Detect → OCR → Transform → Extract → Export
   ```
   Каждый шаг независим и тестируем.

2. **Двухпроходная LLM стратегия:**
   ```python
   # Pass 1: Analyze
   # Pass 2: Extract point-by-point
   ```
   Умная оптимизация для плотных графиков.

3. **Fallback механизмы:**
   ```python
   LLM → CV pipeline → Manual calibration
   ```

4. **Confidence tracking:**
   - Поэлементные метрики (crop, axis, ocr, extraction)
   - Взвешенный overall score
   - Зоны уверенности (high/medium/low)

5. **Warning система:**
   - 12 типов предупреждений
   - Рекомендации по исправлению
   - Enum коды для обработки

---

### ⚠️ Потенциальные улучшения

#### 6.1 Зависимость от Mistral API

**Проблема:** Жесткая зависимость от одного провайдера.

**Рекомендация:** Абстракция LLM провайдера:

```python
class LLMProvider(Protocol):
    def extract_chart(self, image: np.ndarray) -> Dict[str, Any]: ...

class MistralProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...  # Будущее расширение
```

---

#### 6.2 Отсутствие логирования

**Файл:** везде

**Проблема:** Используется `print()` вместо `logging`:
- `api/main.py:93-95` - print в lifespan
- Нет structured logging

**Рекомендация:**
```python
import logging
logger = logging.getLogger(__name__)

# В API
logger.info("Chart2CSV API starting...", extra={
    "version": "1.0.0",
    "env": os.environ.get("ENV", "production")
})
```

---

#### 6.3 Кэширование OCR

**Файл:** `chart2csv/core/cache.py`

✅ Уже реализовано! Disk-based кэширование OCR результатов.

**Потенциальное улучшение:** Добавить TTL и размер кэша:

```python
CACHE_MAX_SIZE = 1000  # Max entries
CACHE_TTL_DAYS = 7     # Auto-cleanup old entries
```

---

## 7. Зависимости (Dependencies) 📦

### Анализ requirements.txt

```python
opencv-python>=4.8.0     # ✅ Актуальная версия
pytesseract>=0.3.10      # ✅ OK
Pillow>=10.0.0           # ✅ Современная версия
numpy>=1.24.0            # ✅ OK
scipy>=1.11.0            # ✅ OK
scikit-image>=0.21.0     # ✅ OK
pypdfium2>=4.0.0         # ⚠️ Не используется в коде?
click>=8.1.0             # ❌ НЕ используется! (было CLI на click?)
pytest>=7.4.0            # ✅ Dev dep
black>=23.0.0            # ✅ Dev dep
mypy>=1.5.0              # ✅ Dev dep
ruff>=0.0.280            # ✅ Dev dep
mistralai>=1.0.0         # ✅ Основная зависимость
```

### ⚠️ Проблемы

1. **click не используется** - CLI построен на `argparse`, не `click`
2. **Нет FastAPI в requirements.txt** - но используется в API!
3. **Нет uvicorn** - нужен для запуска API
4. **Нет pydantic** - используется в API моделях

### Рекомендация

Разделить на:

```text
# requirements.txt (core)
opencv-python>=4.8.0
pytesseract>=0.3.10
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-image>=0.21.0
mistralai>=1.0.0

# requirements-api.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# requirements-dev.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
mypy>=1.5.0
ruff>=0.0.280
```

---

## 8. Производительность (Performance) ⚡

### Потенциальные проблемы

#### 8.1 Синхронная обработка в async endpoint

**Файл:** `api/main.py:183-335`

```python
async def extract_data(...):
    # ...
    result = extract_chart(...)  # 🚨 Блокирующий вызов в async!
```

**Проблема:** CV pipeline блокирует event loop.

**Рекомендация:**
```python
from fastapi import BackgroundTasks
import asyncio

async def extract_data(...):
    result = await asyncio.to_thread(extract_chart, temp_path, ...)
```

---

#### 8.2 Отсутствие connection pooling для Mistral

**Файл:** `chart2csv/core/llm_extraction.py:65`

```python
client = Mistral(api_key=api_key)  # Создается каждый раз!
```

**Рекомендация:**
```python
# Переиспользовать клиент
_mistral_client = None

def get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return _mistral_client
```

---

## 9. Приоритизированный список улучшений

### 🔴 Критические (немедленно)

1. **Безопасность CORS** - исправить `allow_origins=["*"]`
2. **Лицензия в setup.py** - изменить MIT → AGPL-3.0
3. **Добавить FastAPI в requirements** - API не запустится без него
4. **Исправить sys.path в API** - использовать правильную установку
5. **Валидация изображений** - защита от zip-бомб

### 🟡 Важные (в течение недели)

6. **Тестирование** - поднять покрытие до 70%+
7. **setup.py metadata** - исправить author/url
8. **Документация API** - примеры с calibration
9. **Логирование** - заменить print на logging
10. **Async в API** - asyncio.to_thread для CV

### 🟢 Желательные (backlog)

11. **pyproject.toml** - мигрировать с setup.py
12. **pre-commit hooks** - автоматический линтинг
13. **CONTRIBUTING.md** - гайд для разработчиков
14. **CHANGELOG.md** - версионирование
15. **Examples/** - примеры использования
16. **CI/CD** - GitHub Actions
17. **Абстракция LLM** - поддержка других провайдеров
18. **Рефакторинг API** - убрать дублирование

---

## 10. Положительные отзывы 🎉

Несмотря на проблемы, проект демонстрирует:

1. **Профессиональную архитектуру** - четкое разделение concerns
2. **Продуманную систему confidence** - с детальным tracking
3. **Интеллектуальные fallback** - LLM → CV → Manual
4. **Production-ready API** - с rate limiting и error handling
5. **Хорошую типизацию** - type hints везде
6. **Двухпроходную LLM стратегию** - innovative подход
7. **Модульность** - каждый компонент независим

---

## Заключение

**Рекомендация:** Проект готов к использованию, но требует:
1. Исправления критических проблем безопасности
2. Значительного расширения тестов
3. Обновления документации и метаданных

**Оценка готовности к production:** 75%

После исправления критических проблем (1-5 из списка выше) можно считать production-ready.

---

**Конец отчета**
