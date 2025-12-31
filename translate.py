"""
Unified translation CLI - Main entry point for the translation pipeline.
"""

import argparse
import sys
from pathlib import Path

from core.config_loader import ConfigLoader
from core.dataset_loader import DatasetLoader
from core.pipeline import TranslationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Translate datasets using NVIDIA or Fanar APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config.yaml
  python translate.py
  
  # Use custom config
  python translate.py --config my_config.yaml
  
  # Override config settings via CLI
  python translate.py --dataset kaggle:user/dataset --output data/out.csv
  
  # Quick translation of local file
  python translate.py --input data/file.csv --columns text --output data/out.csv --api nvidia
        """
    )
    
    # Configuration
    parser.add_argument("--config", help="Path to YAML config file (default: config.yaml)")
    
    # Dataset options (override config)
    parser.add_argument("--dataset", help="Dataset source (kaggle:user/dataset, huggingface:dataset, or file path)")
    parser.add_argument("--input", help="Alias for --dataset (for local files)")
    parser.add_argument("--subset", help="Configuration/subset name for Hugging Face datasets (e.g. toxicchat0124)")
    parser.add_argument("--file-name", help="Specific file within Kaggle dataset")
    parser.add_argument("--columns", nargs="+", help="Columns to translate (or 'auto')")
    
    # Translation options
    parser.add_argument("--api", choices=["nvidia", "fanar"], help="Primary API to use")
    parser.add_argument("--fallback", action="store_true", help="Enable fallback to secondary API")
    parser.add_argument("--no-normalize", action="store_true", help="Disable provider term normalization")
    
    # Output options
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["csv", "json", "parquet"], help="Output format")
    
    # Performance options
    parser.add_argument("--limit", type=int, help="Limit number of rows (for testing)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--checkpoint-dir", help="Custom checkpoint directory (default: checkpoints/)")
    parser.add_argument("--keep-columns", nargs="+", help="Specific columns to keep in output (default: all)")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        loader = ConfigLoader(args.config)
        config = loader.load()
        loader.validate()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file with your API keys (see .env.example)")
        print("2. Set up config.yaml (or use --config)")
        sys.exit(1)
    
    # Override config with CLI arguments
    dataset_source = args.input or args.dataset or config['dataset']['source']
    if not dataset_source:
        print("Error: No dataset specified. Use --dataset, --input, or set in config.yaml")
        sys.exit(1)
    
    columns = args.columns if args.columns else config['dataset']['columns_to_translate']
    primary_api = args.api or config['translation']['primary_api']
    enable_fallback = args.fallback or config['translation']['enable_fallback']
    normalize_terms = not args.no_normalize and config['translation']['normalize_provider_terms']
    output_path = args.output or config['output']['path']
    output_format = args.format or config['output']['format']
    
    # Load dataset
    print(f"Loading dataset: {dataset_source}")
    try:
        dataset_loader = DatasetLoader()
        if args.file_name:
            df = dataset_loader._load_kaggle(dataset_source, args.file_name)
        else:
            df = dataset_loader.load_dataset(dataset_source, config_name=args.subset)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Limit rows if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} rows")
    
    # Determine columns to translate
    if columns == "auto" or columns == ["auto"]:
        columns = dataset_loader.get_text_columns()
        print(f"Auto-detected columns: {columns}")
    
    if not columns:
        print("Error: No columns to translate. Use --columns or set in config.yaml")
        sys.exit(1)
    
    # Initialize pipeline
    print(f"\nTranslation settings:")
    print(f"  Primary API: {primary_api}")
    print(f"  Fallback enabled: {enable_fallback}")
    print(f"  Columns: {columns}")
    print(f"  Rate limiting: {config['apis'][primary_api]['rate_limit_rpm']} RPM")
    
    pipeline = TranslationPipeline(
        columns_to_translate=columns,
        primary_api=primary_api,
        enable_fallback=enable_fallback,
        normalize_provider_terms=normalize_terms,
        rate_limit_rpm=config['apis'][primary_api]['rate_limit_rpm'] if config['retry']['respect_rate_limits'] else None,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None
    )
    
    # Translate
    print("\nStarting translation...")
    try:
        result = pipeline.translate_dataset(
            df,
            resume_from_checkpoint=args.resume
        )
    except KeyboardInterrupt:
        print("\nTranslation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during translation: {e}")
        sys.exit(1)
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter columns if --keep-columns specified
    if args.keep_columns:
        # Always include translated columns
        translated_cols = [f"{col}_ar" for col in columns] + [f"{col}_api" for col in columns]
        # Add user-specified columns
        keep_cols = args.keep_columns + translated_cols
        # Filter to only existing columns
        available_cols = [col for col in keep_cols if col in result.columns]
        result = result[available_cols]
        print(f"Keeping columns: {available_cols}")
    
    print(f"\nSaving to {output_path}...")
    if output_format == "csv" or str(output_path).endswith('.csv'):
        result.to_csv(output_path, index=False, encoding='utf-8')
    elif output_format == "json" or str(output_path).endswith('.json'):
        result.to_json(output_path, orient='records', indent=2, force_ascii=False)
    elif output_format == "parquet" or str(output_path).endswith('.parquet'):
        result.to_parquet(output_path)
    else:
        result.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Translation complete: {output_path}")
    
    # Save statistics
    if config['output']['save_statistics']:
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        pipeline.save_statistics(stats_path)
        print(f"✓ Statistics saved: {stats_path}")


if __name__ == "__main__":
    main()
