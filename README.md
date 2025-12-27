# Dataset Translation Pipeline

**Configurable English-to-Arabic translation pipeline** for datasets from Kaggle, HuggingFace, or local files. Uses NVIDIA or Fanar APIs with automatic rate limiting, checkpointing, and quality validation.

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd DatasetTranslation
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Required: At least one API key
NVIDIA_API_KEY=your_nvidia_key_here
FANAR_API_KEY=your_fanar_key_here

# Optional: For Kaggle datasets
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

**Get API Keys:**
- **NVIDIA**: [build.nvidia.com](https://build.nvidia.com/)
- **Fanar**: [fanar.qa](https://fanar.qa/)
- **Kaggle**: [kaggle.com/settings](https://www.kaggle.com/settings)

### 3. Configure & Run

Edit `config.yaml` to specify your dataset and settings, then:

```bash
python translate.py
```

## 📖 Usage

### Using Default Config

```bash
# Uses config.yaml
python translate.py
```

### Override via CLI

```bash
# Translate Kaggle dataset
python translate.py --dataset kaggle:user/dataset-name --output data/out.csv

# Translate HuggingFace dataset
python translate.py --dataset user/dataset-name --columns prompt

# Translate local CSV
python translate.py --input data/file.csv --columns text description --api nvidia

# Enable fallback API
python translate.py --dataset my-dataset --api nvidia --fallback

# Test with limited rows
python translate.py --dataset my-dataset --limit 10
```

### Using Custom Config

```bash
python translate.py --config examples/kaggle_example.yaml
```

## ⚙️ Configuration

### config.yaml Structure

```yaml
dataset:
  source: "kaggle:user/dataset"  # or "huggingface:dataset" or "path/file.csv"
  columns_to_translate: auto     # or ["column1", "column2"]

translation:
  primary_api: nvidia            # nvidia or fanar
  enable_fallback: false         # Use secondary API if primary fails
  normalize_provider_terms: true # Replace ChatGPT/OpenAI with generic terms

apis:
  nvidia:
    rate_limit_rpm: 40           # Requests per minute
  fanar:
    rate_limit_rpm: 40

output:
  path: data/output.csv
  format: csv                    # csv, json, or parquet
```

See `examples/` for complete configuration examples.

## 📁 Project Structure

```
DatasetTranslation/
├── translate.py              # Main CLI entry point
├── config.yaml               # Default configuration
├── .env                      # API keys (create from .env.example)
├── examples/                 # Example configurations
│   ├── kaggle_example.yaml
│   ├── huggingface_example.yaml
│   └── local_csv_example.yaml
├── core/                     # Core pipeline components
│   ├── config_loader.py      # YAML & env config loader
│   ├── dataset_loader.py     # Multi-source dataset loader
│   ├── pipeline.py           # Translation pipeline
│   ├── rate_limiter.py       # API rate limiting
│   ├── preprocessing.py      # Text preprocessing
│   └── postprocessing.py     # Translation validation
├── apis/                     # API clients
│   ├── nvidia.py
│   └── fanar.py
├── data/                     # Output datasets
├── logs/                     # Execution logs
└── checkpoints/              # Progress checkpoints
```

## ✨ Features

- **Multi-Source Support**: Kaggle, HuggingFace, CSV, JSON, JSONL, Parquet
- **Dual-API Support**: NVIDIA (primary) + Fanar (optional fallback)
- **Rate Limiting**: Automatic throttling (40 RPM for both APIs)
- **Checkpointing**: Resume interrupted translations
- **Quality Validation**: Arabic character detection, length checks
- **Provider Normalization**: ChatGPT/OpenAI → generic terms
- **Flexible Configuration**: YAML config + CLI overrides
- **Comprehensive Logging**: Detailed logs and statistics

## 🔧 Advanced Usage

### Resume from Checkpoint

```bash
python translate.py --resume
```

### Disable Rate Limiting

Edit `config.yaml`:
```yaml
retry:
  respect_rate_limits: false
```

### Custom API Parameters

Edit `config.yaml`:
```yaml
apis:
  nvidia:
    temperature: 0.2
    max_tokens: 1024
```

## 📊 Output Format

The translated CSV includes:
- All original columns (if `preserve_all_columns: true`)
- `{column}_ar`: Arabic translation
- `{column}_api`: API used (nvidia/fanar)

Example:
```csv
id,prompt,prompt_ar,prompt_api
1,"Ignore instructions","تجاهل التعليمات",nvidia
```

## 🐛 Troubleshooting

### "NVIDIA_API_KEY not found"
- Ensure `.env` file exists with your API key
- Check `.env` is in the project root directory

### "No columns to translate"
- Specify columns in `config.yaml` or use `--columns`
- Use `columns_to_translate: auto` to auto-detect

### Rate Limit Errors
- Pipeline automatically throttles to 40 RPM
- Check logs for rate limit wait times

### Kaggle Authentication Failed
- Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`
- Or place `kaggle.json` in `~/.kaggle/`

## 📝 Examples

See `examples/` directory for:
- `kaggle_example.yaml` - Kaggle dataset translation
- `huggingface_example.yaml` - HuggingFace dataset
- `local_csv_example.yaml` - Local CSV file

## 🔒 Security

- **Never commit `.env`** - It's in `.gitignore` by default
- API keys are loaded from environment variables only
- Share `.env.example` as a template for colleagues

## 📄 License

For research purposes in Machine Translation, LLM Security, and Arabic NLP in general.

<!-- ## 🤝 Sharing with Colleagues

1. Share the repository (without `.env`)
2. Colleague creates their own `.env` from `.env.example`
3. Colleague edits `config.yaml` for their dataset
4. Run `python translate.py` -->
