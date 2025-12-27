"""
Translation quality validation and post-processing utilities.
"""

import re
from typing import Dict, Tuple
from core.config import (
    MIN_TRANSLATION_LENGTH,
    MAX_LENGTH_RATIO,
    MIN_LENGTH_RATIO,
    MIN_ARABIC_CHAR_RATIO
)


def has_arabic_chars(text: str) -> bool:
    """
    Check if text contains Arabic characters.
    
    Args:
        text: Input text
        
    Returns:
        True if text contains Arabic characters
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    return bool(arabic_pattern.search(text))


def get_arabic_char_ratio(text: str) -> float:
    """
    Calculate the ratio of Arabic characters in text.
    
    Args:
        text: Input text
        
    Returns:
        Ratio of Arabic characters (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    arabic_chars = len(arabic_pattern.findall(text))
    total_chars = len(re.sub(r'\s', '', text))  # Exclude whitespace
    
    return arabic_chars / max(total_chars, 1)


def has_mojibake(text: str) -> bool:
    """
    Detect potential encoding issues (mojibake).
    
    Args:
        text: Input text
        
    Returns:
        True if encoding issues detected
    """
    # Common mojibake patterns
    mojibake_patterns = [
        r'Ã©|Ã¨|Ã |Ã§',  # Common UTF-8 to Latin-1 issues
        r'â€™|â€œ|â€�',  # Smart quotes issues
        r'\ufffd',  # Replacement character
    ]
    
    for pattern in mojibake_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def validate_translation(original: str, translated: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate translation quality using multiple checks.
    
    Args:
        original: Original English text
        translated: Translated Arabic text
        
    Returns:
        Tuple of (is_valid, checks_dict)
    """
    checks = {}
    
    # Check 1: Not empty
    checks["not_empty"] = len(translated.strip()) >= MIN_TRANSLATION_LENGTH
    
    # Check 2: Contains Arabic characters
    checks["has_arabic"] = has_arabic_chars(translated)
    
    # Check 3: Sufficient Arabic character ratio
    checks["arabic_ratio_ok"] = get_arabic_char_ratio(translated) >= MIN_ARABIC_CHAR_RATIO
    
    # Check 4: Reasonable length ratio
    if len(original) > 0:
        length_ratio = len(translated) / len(original)
        checks["reasonable_length"] = MIN_LENGTH_RATIO <= length_ratio <= MAX_LENGTH_RATIO
    else:
        checks["reasonable_length"] = True
    
    # Check 5: No encoding errors
    checks["no_encoding_errors"] = not has_mojibake(translated)
    
    # Overall validation
    is_valid = all(checks.values())
    
    return is_valid, checks


def clean_arabic_text(text: str) -> str:
    """
    Clean and normalize Arabic text.
    
    Args:
        text: Arabic text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize Arabic characters
    # Replace different forms of Alef
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize Teh Marbuta
    text = re.sub(r'ة', 'ة', text)
    
    # Remove excessive diacritics (keep text readable but clean)
    # This is optional - comment out if you want to preserve all diacritics
    # text = re.sub(r'[\u064B-\u0652]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def postprocess_translation(translated: str) -> str:
    """
    Complete post-processing pipeline for translated text.
    
    Args:
        translated: Translated Arabic text
        
    Returns:
        Post-processed translation
    """
    # Clean Arabic text
    text = clean_arabic_text(translated)
    
    return text


if __name__ == "__main__":
    # Test examples
    test_cases = [
        ("Hello, how are you?", "مرحبا، كيف حالك؟"),
        ("This is a test", "هذا اختبار"),
        ("Short", "قصير جداً جداً جداً جداً جداً"),  # Too long
        ("Some text", ""),  # Empty translation
        ("Arabic test", "This is still English"),  # No Arabic
    ]
    
    print("=== Translation Validation Test ===\n")
    for original, translated in test_cases:
        is_valid, checks = validate_translation(original, translated)
        print(f"Original:    {original}")
        print(f"Translated:  {translated}")
        print(f"Valid:       {is_valid}")
        print(f"Checks:      {checks}\n")
