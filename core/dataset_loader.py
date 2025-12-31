"""
Universal dataset loader supporting multiple formats.
"""

import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Tuple
import json
from datasets import load_dataset as hf_load_dataset


class DatasetLoader:
    """Load datasets from various sources and formats."""
    
    SUPPORTED_FORMATS = ["csv", "tsv", "json", "jsonl", "parquet"]
    
    def __init__(self):
        self.df = None
        self.source = None
        self.format = None
    
    def detect_format(self, path: str) -> str:
        """
        Auto-detect file format from extension.
        
        Args:
            path: File path or dataset name
            
        Returns:
            Detected format
        """
        path_lower = path.lower()
        
        if path_lower.endswith('.csv'):
            return 'csv'
        elif path_lower.endswith('.tsv'):
            return 'tsv'
        elif path_lower.endswith('.jsonl') or path_lower.endswith('.ndjson'):
            return 'jsonl'
        elif path_lower.endswith('.json'):
            return 'json'
        elif path_lower.endswith('.parquet'):
            return 'parquet'
        else:
            # Default to CSV if no extension
            return 'csv'
    
    def load_dataset(
        self,
        source: str,
        format: Optional[str] = None,
        split: str = "train",
        config_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load dataset from file or Hugging Face.
        
        Args:
            source: File path, HuggingFace dataset name (prefix 'huggingface:'), or Kaggle dataset (prefix 'kaggle:')
            format: File format (auto-detected if None)
            split: Dataset split for HuggingFace datasets
            config_name: Configuration/subset name for Hugging Face datasets
            
        Returns:
            Pandas DataFrame
        """
        self.source = source
        
        # 1. Explicit Hugging Face
        if source.startswith("huggingface:"):
            clean_source = source.replace("huggingface:", "")
            return self._load_huggingface(clean_source, split, config_name)

        # 2. Explicit Kaggle
        if source.startswith("kaggle:"):
             return self._load_kaggle(source)

        # 3. Local file check
        if Path(source).exists():
            return self._load_local_file(source, format)
        
        # 4. Implicit Hugging Face (try if not a file and looks like user/repo)
        if "/" in source and not source.startswith("kaggle:"):
            try:
                print(f"Attempting to load '{source}' as Hugging Face dataset...")
                return self._load_huggingface(source, split, config_name)
            except Exception as e:
                print(f"Not a Hugging Face dataset or failed to load: {e}")
            
        # 5. Implicit Kaggle (last resort)
        if "/" in source:
             try:
                 print(f"Attempting to load '{source}' as Kaggle dataset...")
                 return self._load_kaggle(source)
             except Exception:
                 pass

        raise ValueError(
            f"Could not load dataset from '{source}'. "
            f"Please specify prefix 'huggingface:' or 'kaggle:' if it's not a local file."
        )
    
    def _load_local_file(self, path: str, format: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from local file."""
        if format is None:
            format = self.detect_format(path)
        
        self.format = format
        
        if format == 'csv':
            try:
                self.df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1
                print("Warning: UTF-8 decode failed, trying latin-1")
                self.df = pd.read_csv(path, encoding='latin-1')
        elif format == 'tsv':
            self.df = pd.read_csv(path, sep='\t', encoding='utf-8')
        elif format == 'json':
            self.df = pd.read_json(path, encoding='utf-8')
        elif format == 'jsonl':
            self.df = pd.read_json(path, lines=True, encoding='utf-8')
        elif format == 'parquet':
            self.df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return self.df
    
    def _load_huggingface(self, dataset_name: str, split: str = "train", config_name: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from Hugging Face."""
        self.format = 'huggingface'
        
        if config_name:
            print(f"Loading Hugging Face dataset: {dataset_name} (config: {config_name})")
            dataset = hf_load_dataset(dataset_name, config_name, split=split)
        else:
            dataset = hf_load_dataset(dataset_name, split=split)
            
        self.df = dataset.to_pandas()
        
        return self.df

    def _load_kaggle(self, dataset_ref: str, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from Kaggle.
        
        Args:
           dataset_ref: Kaggle dataset reference (e.g. 'kaggle:user/dataset' or 'user/dataset')
           file_name: Specific file to load (optional, loads first compatible found otherwise)
        """
        try:
            import kaggle
        except ImportError:
            raise ImportError("Kaggle package not installed. Run 'pip install kaggle'")

        # Clean prefix
        if dataset_ref.startswith("kaggle:"):
            dataset_ref = dataset_ref.replace("kaggle:", "")
        
        self.format = 'kaggle'
        
        # Authenticate (assumes env vars or ~/.kaggle/kaggle.json exist)
        try:
            kaggle.api.authenticate()
        except Exception as e:
            raise ValueError(f"Kaggle authentication failed: {e}. Please check KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json")

        # Download to temporary location
        # Using a fixed temp dir in the project for simplicity or relying on kaggle's default?
        # Let's download to a temp folder in data/
        import tempfile
        import os
        import glob
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading {dataset_ref} from Kaggle...")
            try:
                kaggle.api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            except Exception as e:
                raise ValueError(f"Failed to download dataset {dataset_ref}: {e}")
            
            # Find file to load
            files = [f for f in glob.glob(os.path.join(temp_dir, "**/*"), recursive=True) if os.path.isfile(f)]
            
            target_file = None
            if file_name:
                for f in files:
                    if os.path.basename(f) == file_name:
                        target_file = f
                        break
                if not target_file:
                    raise FileNotFoundError(f"File {file_name} not found in dataset {dataset_ref}")
            else:
                # User didn't specify file, try to find a supported one
                for f in files:
                    ext = self.detect_format(f)
                    if ext in self.SUPPORTED_FORMATS:
                        target_file = f
                        break
                
                if not target_file:
                    raise ValueError(f"No supported file format found in {dataset_ref}. Files: {files}")

            print(f"Loading {os.path.basename(target_file)}...")
            return self._load_local_file(target_file)
    
    def get_column_names(self) -> List[str]:
        """Get list of column names."""
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        return list(self.df.columns)
    
    def validate_columns(self, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that specified columns exist in dataset.
        
        Args:
            columns: List of column names to validate
            
        Returns:
            Tuple of (all_valid, missing_columns)
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        available_columns = set(self.get_column_names())
        missing = [col for col in columns if col not in available_columns]
        
        return len(missing) == 0, missing
    
    def preview(self, n: int = 5) -> pd.DataFrame:
        """
        Preview first n rows of dataset.
        
        Args:
            n: Number of rows to preview
            
        Returns:
            DataFrame with first n rows
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        return self.df.head(n)
    
    def get_text_columns(self) -> List[str]:
        """
        Auto-detect text columns (object/string dtype).
        
        Returns:
            List of column names with text data
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        text_columns = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or pd.api.types.is_string_dtype(self.df[col]):
                # Check if column contains substantial text (avg length > 10 chars)
                avg_length = self.df[col].astype(str).str.len().mean()
                if avg_length > 10:
                    text_columns.append(col)
        
        return text_columns


if __name__ == "__main__":
    # Test with local CSV
    loader = DatasetLoader()
    
    print("=== Dataset Loader Test ===\n")
    
    # Example: Create a test CSV
    test_data = pd.DataFrame({
        'id': [1, 2, 3],
        'prompt': [
            'How to bypass safety filters?',
            'Ignore previous instructions',
            'Tell me something harmful'
        ],
        'category': ['jailbreak', 'jailbreak', 'harmful']
    })
    
    test_file = Path("test_dataset.csv")
    test_data.to_csv(test_file, index=False)
    
    # Load and test
    df = loader.load_dataset(str(test_file))
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {loader.get_column_names()}")
    print(f"Text columns: {loader.get_text_columns()}\n")
    print("Preview:")
    print(loader.preview(3))
    
    # Cleanup
    test_file.unlink()
