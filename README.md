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

# Translate HuggingFace dataset with specific configuration/subset
python translate.py --dataset lmsys/toxic-chat --subset toxicchat0124 --columns user_input model_output

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

You can configure the pipeline in **two ways** (or combine both):

1. **config.yaml file** - Set defaults for repeated use
2. **CLI arguments** - Override config.yaml for one-time changes

**Priority**: CLI arguments > config.yaml > built-in defaults

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

### CLI Arguments Reference

All CLI arguments override corresponding `config.yaml` settings:

#### Configuration
- `--config PATH` - Path to custom YAML config file (default: `config.yaml`)

#### Dataset Options
- `--dataset SOURCE` - Dataset source:
  - Kaggle: `kaggle:user/dataset-name`
  - HuggingFace: `user/dataset-name` or `huggingface:user/dataset`
  - Local file: `path/to/file.csv`
- `--input PATH` - Alias for `--dataset` (for local files)
- `--subset NAME` - Configuration/subset name for Hugging Face datasets
- `--file-name NAME` - Specific file within Kaggle dataset (if multiple files)
- `--columns COL [COL ...]` - Columns to translate (space-separated) or `auto` for auto-detection

#### Translation Options
- `--api {nvidia,fanar}` - Primary translation API
- `--fallback` - Enable fallback to secondary API if primary fails (default: disabled)
- `--no-normalize` - Disable ChatGPT/OpenAI term normalization

#### Output Options
- `--output PATH` - Output file path (default: `data/translated_output.csv`)
- `--format {csv,json,parquet}` - Output format (auto-detected from file extension)
- `--keep-columns COL [COL ...]` - Specific original columns to keep in output (default: all). Translated columns always included.

#### Performance Options
- `--limit N` - Translate only first N rows (useful for testing)
- `--no-checkpoint` - Disable checkpointing
- `--resume` - Resume from last checkpoint
- `--checkpoint-dir PATH` - Custom checkpoint directory for parallel translations (default: `checkpoints/`)

### config.yaml Reference

Complete configuration file structure:

```yaml
# Dataset Configuration
dataset:
  source: null                    # Required: dataset source
  file_name: null                 # Optional: specific file in Kaggle dataset
  columns_to_translate: auto      # "auto" or list of column names

# Translation Settings
translation:
  primary_api: nvidia             # "nvidia" or "fanar"
  enable_fallback: false          # Use secondary API if primary fails
  normalize_provider_terms: true  # Replace ChatGPT/OpenAI with generic terms
  preserve_all_columns: true      # Keep all original columns in output

# API Configuration (credentials from .env)
apis:
  nvidia:
    model: nvidia/riva-translate-4b-instruct-v1.1
    temperature: 0.1              # 0.0-1.0, lower = more consistent
    max_tokens: 512
    top_p: 0.7
    rate_limit_rpm: 40            # Requests per minute
  fanar:
    model: Fanar
    temperature: 0.3
    max_tokens: 1024
    rate_limit_rpm: 40

# Retry & Performance
retry:
  max_retries: 3                  # Retries per API call
  backoff_multiplier: 2           # Exponential backoff multiplier
  request_timeout: 30             # Seconds
  respect_rate_limits: true       # Auto-throttle to stay within limits

# Checkpointing
checkpoint:
  enabled: true
  interval: 50                    # Save progress every N rows
  resume_from_last: false

# Output
output:
  path: data/translated_output.csv
  format: csv                     # csv, json, or parquet
  save_statistics: true           # Save translation stats to JSON
```

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


## 📄 License

For research purposes in Machine Translation, LLM Security, and Arabic NLP in general.
