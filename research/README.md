# Исследование детекторов SunoCheck

*English: research scripts and lab notes behind the detectors (in Russian). Start with `REPORT.md`; every tested hypothesis, including the rejected ones, is in `HYPOTHESES.md`.*

Здесь лежит то, на чём построены детекторы приложения:

- `REPORT.md` — сводка: что нашли и какие подходы работают;
- `HYPOTHESES.md` — журнал всех проверенных идей с цифрами, включая отвергнутые;
- скрипты `*.py` — загрузка корпуса, подготовка спектров, эксперименты.

Данные, кеши и метаданные корпуса в репозиторий не входят (около 8 ГБ). Ниже — как собрать их заново.

## Окружение

```powershell
uv sync --all-groups
```

Группа `research` добавляет `pyarrow` и `scikit-learn`. Для вариантов корпуса (кодеки, смена скорости и тона) нужен `ffmpeg` с фильтрами `rubberband` и `afir` в `PATH`.

## Корпус

| Источник | Что | Куда |
|---|---|---|
| [Kukedlc/suno-ai-music-dataset](https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset) | Suno v2–v5.5, первые ~60 с трека | `research/data/suno` |
| [FMA medium](https://huggingface.co/datasets/benjamin-paine/free-music-archive-medium) | люди, клипы по 30 с | `research/data/fma` |
| [MTG-Jamendo](https://huggingface.co/datasets/rkstgr/mtg-jamendo) | люди, полные треки | `research/data/jamendo` |
| свои файлы | дополнительные треки Suno | `samples/suno`, `samples/sudo` |

1. Сохраните CSV метаданных датасета Kukedlc как `research/meta/kukedlc_metadata.csv`. Нужны столбцы `id`, `title`, `major_model_version`, `is_instrumental`, `file_name`.
2. Загрузите аудио:

   ```powershell
   uv run python research/fetch_data.py suno
   uv run python research/fetch_data.py fma
   uv run python research/fetch_data.py jamendo
   ```

3. Посчитайте спектры для нужных вариантов: `uv run python research/prepare.py <workers> all <variant[,...]> 1`. Варианты перечислены в `corpus.py` (`VARIANTS`).

У каждого скрипта в начале файла есть строка `Usage:` с параметрами запуска.

## Основные скрипты

| Скрипт | Что делает |
|---|---|
| `corpus.py`, `prepare.py` | корпус, варианты кодирования, кеш спектров |
| `experiments.py` | fakeprint: окна, полосы, кодеки, AUROC и TPR при FPR 1 % / 0.1 % |
| `coherent_*.py`, `scan_*.py` | когерентная гребёнка 200 Гц и поиск базы при смене скорости |
| `comb_growing*.py` | растущее окно гребёнки (детектор приложения) |
| `semitone_*.py`, `shifted_model.py`, `shift_tool_match.py` | Suno с пониженным тоном; `shifted_model.py fit` обучает `src/sunocheck/models/shifted_weights.npz` |
| `warp.py`, `warp_model.py` | чтение спектра в точках r·f (отмена сдвига) |

## Лицензии данных

Аудио датасетов распространяется на условиях их авторов (FMA и MTG-Jamendo — Creative Commons по трекам). Поэтому в репозиторий не входят ни аудио, ни производные кеши. Веса моделей в `src/sunocheck/models` — числовые коэффициенты логистической регрессии, само аудио в них не хранится.
