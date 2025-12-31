"""
Main translation pipeline with retry logic, checkpointing, and dual-API support.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import json

from core.config import (
    MAX_RETRIES,
    BACKOFF_MULTIPLIER,
    REQUEST_TIMEOUT,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_DIR
)
from core.preprocessing import preprocess_for_translation
from core.postprocessing import validate_translation, postprocess_translation
from apis.nvidia import translate_en_to_ar as nvidia_translate
from apis.fanar import FanarClient


class TranslationPipeline:
    """
    Translation pipeline with retry logic and dual-API support.
    """
    
    def __init__(
        self,
        columns_to_translate: List[str],
        primary_api: str = "nvidia",
        enable_fallback: bool = False,
        preserve_all_columns: bool = True,
        normalize_provider_terms: bool = True,
        checkpoint_dir: Optional[Path] = None,
        rate_limit_rpm: Optional[int] = 40
    ):
        """
        Initialize translation pipeline.
        
        Args:
            columns_to_translate: List of column names to translate
            primary_api: Primary API to use ("nvidia" or "fanar")
            enable_fallback: Use secondary API if primary fails (default: False)
            preserve_all_columns: Keep all original columns in output
            normalize_provider_terms: Normalize provider-specific terms before translation
            checkpoint_dir: Directory for checkpoint files (default: checkpoints/)
            rate_limit_rpm: Requests per minute limit (None to disable)
        """
        self.columns_to_translate = columns_to_translate
        self.primary_api = primary_api
        self.enable_fallback = enable_fallback
        self.preserve_all_columns = preserve_all_columns
        self.normalize_provider_terms = normalize_provider_terms
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        
        # Initialize rate limiter
        from core.rate_limiter import RateLimiter
        self.rate_limiter = RateLimiter(rate_limit_rpm) if rate_limit_rpm else None
        
        # Initialize Fanar client
        from core.config import FANAR_API_KEY
        self.fanar_client = FanarClient(FANAR_API_KEY)
        
        # Statistics
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'nvidia_used': 0,
            'fanar_used': 0,
            'start_time': None,
            'end_time': None,
            'failed_rows': []
        }
        
        # Setup logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('TranslationPipeline')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        from core.config import LOGS_DIR
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"translation_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        from core.config import LOG_FORMAT, LOG_DATE_FORMAT
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def translate_with_retry(
        self,
        text: str,
        max_retries: int = MAX_RETRIES
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Translate text with retry logic and API fallback.
        
        Args:
            text: Text to translate
            max_retries: Maximum retry attempts per API
            
        Returns:
            Tuple of (translation, api_used, error)
        """
        # Preprocess text
        processed_text = preprocess_for_translation(
            text,
            normalize_terms=self.normalize_provider_terms
        )
        
        # Determine API order (only use fallback if enabled)
        if self.primary_api == "nvidia":
            apis = [("nvidia", nvidia_translate)]
            if self.enable_fallback:
                apis.append(("fanar", self.fanar_client.translate_en_to_ar))
        else:
            apis = [("fanar", self.fanar_client.translate_en_to_ar)]
            if self.enable_fallback:
                apis.append(("nvidia", nvidia_translate))
        
        # Try each API
        for api_name, api_func in apis:
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"Attempting {api_name} (attempt {attempt + 1}/{max_retries})")
                    
                    # Rate limiting
                    if self.rate_limiter:
                        wait_time = self.rate_limiter.get_wait_time()
                        if wait_time > 0:
                            self.logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                        self.rate_limiter.acquire()
                    
                    # Call API
                    result = api_func(processed_text, timeout=REQUEST_TIMEOUT)
                    
                    if result['success']:
                        translation = result['translation']
                        
                        # Validate translation
                        is_valid, checks = validate_translation(text, translation)
                        
                        if is_valid:
                            # Post-process
                            final_translation = postprocess_translation(translation)
                            return final_translation, api_name, None
                        else:
                            self.logger.warning(
                                f"{api_name} translation failed validation: {checks}"
                            )
                    else:
                        self.logger.warning(
                            f"{api_name} API error: {result['error']}"
                        )
                
                except Exception as e:
                    self.logger.error(f"{api_name} exception: {str(e)}")
                
                # Exponential backoff
                if attempt < max_retries - 1:
                    sleep_time = BACKOFF_MULTIPLIER ** attempt
                    self.logger.debug(f"Backing off for {sleep_time}s")
                    time.sleep(sleep_time)
        
        # All attempts failed
        return None, "failed", "All API attempts exhausted"
    
    def translate_dataset(
        self,
        df: pd.DataFrame,
        resume_from_checkpoint: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Translate entire dataset with checkpointing.
        
        Args:
            df: Input DataFrame
            resume_from_checkpoint: Resume from last checkpoint
            checkpoint_name: Specific checkpoint file to resume from
            
        Returns:
            DataFrame with translated columns
        """
        self.stats['start_time'] = datetime.now()
        self.stats['total'] = len(df)
        
        self.logger.info(f"Starting translation of {len(df)} rows")
        self.logger.info(f"Columns to translate: {self.columns_to_translate}")
        self.logger.info(f"Primary API: {self.primary_api}")
        
        # Create result DataFrame
        result_df = df.copy() if self.preserve_all_columns else pd.DataFrame()
        
        # Add translated columns
        for col in self.columns_to_translate:
            result_df[f"{col}_ar"] = None
            result_df[f"{col}_api"] = None
        
        # Resume from checkpoint if requested
        start_idx = 0
        if resume_from_checkpoint:
            start_idx = self._load_checkpoint(result_df, checkpoint_name)
        
        # Translate each row
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            
            self.logger.info(f"Processing row {idx + 1}/{len(df)}")
            
            # Translate each column
            for col in self.columns_to_translate:
                text = str(row[col])
                
                translation, api_used, error = self.translate_with_retry(text)
                
                if translation:
                    result_df.at[idx, f"{col}_ar"] = translation
                    result_df.at[idx, f"{col}_api"] = api_used
                    self.stats['successful'] += 1
                    
                    if api_used == "nvidia":
                        self.stats['nvidia_used'] += 1
                    elif api_used == "fanar":
                        self.stats['fanar_used'] += 1
                else:
                    self.stats['failed'] += 1
                    self.stats['failed_rows'].append({
                        'row_idx': idx,
                        'column': col,
                        'error': error
                    })
                    self.logger.error(f"Failed to translate row {idx}, column {col}: {error}")
            
            # Checkpoint periodically
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(result_df, idx + 1)
        
        self.stats['end_time'] = datetime.now()
        self._log_statistics()
        
        return result_df
    
    def _save_checkpoint(self, df: pd.DataFrame, current_idx: int):
        """Save partial checkpoint to disk (only processed rows)."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{current_idx}.csv"
        
        # Only save rows up to current_idx to save space
        partial_df = df.iloc[:current_idx]
        partial_df.to_csv(checkpoint_file, index=False, encoding='utf-8')
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file} ({len(partial_df)} rows)")
    
    def _load_checkpoint(
        self,
        result_df: pd.DataFrame,
        checkpoint_name: Optional[str] = None
    ) -> int:
        """
        Load from checkpoint and return starting index.
        
        Returns:
            Index to resume from
        """
        if checkpoint_name:
            checkpoint_file = self.checkpoint_dir / checkpoint_name
        else:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.csv"))
            if not checkpoints:
                self.logger.info("No checkpoints found, starting from beginning")
                return 0
            checkpoint_file = checkpoints[-1]
        
        if checkpoint_file.exists():
            checkpoint_df = pd.read_csv(checkpoint_file, encoding='utf-8')
            
            # Copy checkpoint data to result
            for col in checkpoint_df.columns:
                if col in result_df.columns:
                    result_df[col] = checkpoint_df[col]
            
            # Find last completed row
            resume_idx = int(checkpoint_file.stem.split('_')[1])
            self.logger.info(f"Resuming from checkpoint: {checkpoint_file} (row {resume_idx})")
            return resume_idx
        
        return 0
    
    def _log_statistics(self):
        """Log final statistics."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info("=" * 60)
        self.logger.info("TRANSLATION STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total rows: {self.stats['total']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"NVIDIA API used: {self.stats['nvidia_used']}")
        self.logger.info(f"Fanar API used: {self.stats['fanar_used']}")
        self.logger.info(f"Duration: {duration:.2f}s")
        self.logger.info(f"Average time per row: {duration / max(self.stats['total'], 1):.2f}s")
        
        if self.stats['failed_rows']:
            self.logger.warning(f"Failed rows: {len(self.stats['failed_rows'])}")
            for failed in self.stats['failed_rows'][:10]:  # Show first 10
                self.logger.warning(f"  Row {failed['row_idx']}, Column {failed['column']}: {failed['error']}")
    
    def save_statistics(self, output_path: Path):
        """Save statistics to JSON file."""
        stats_dict = self.stats.copy()
        stats_dict['start_time'] = stats_dict['start_time'].isoformat() if stats_dict['start_time'] else None
        stats_dict['end_time'] = stats_dict['end_time'].isoformat() if stats_dict['end_time'] else None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Statistics saved to: {output_path}")
