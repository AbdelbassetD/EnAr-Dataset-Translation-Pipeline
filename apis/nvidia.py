# NVIDIA Riva Translate 4B Instruct v1.1 - English to Arabic Translation
# Using the official prompt template from NVIDIA documentation

from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"  # Get from https://build.nvidia.com
)

def translate_en_to_ar(text, timeout=30):
    """
    Translate English text to Arabic using NVIDIA Riva Translate 4B Instruct v1.1
    
    Args:
        text: English text to translate
        timeout: Request timeout in seconds
    
    Returns:
        dict: {
            'success': bool,
            'translation': str or None,
            'error': str or None
        }
    """
    
    # Create messages using the official prompt template
    messages = [
        {
            "role": "system",
            "content": "You are an expert at translating text from English to Arabic."
        },
        {
            "role": "user",
            "content": f"What is the Arabic translation of the sentence: {text}?"
        }
    ]

    try:
        # Make API call
        completion = client.chat.completions.create(
            model="nvidia/riva-translate-4b-instruct-v1.1",
            messages=messages,
            temperature=0.1,  # Lower for more consistent translations
            max_tokens=512,
            top_p=0.7,
            stream=False,
            timeout=timeout
        )
        
        translation = completion.choices[0].message.content
        
        return {
            'success': True,
            'translation': translation,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'translation': None,
            'error': str(e)
        }


def translate_batch_en_to_ar(texts, timeout=30):
    """
    Translate multiple English texts to Arabic
    
    Args:
        texts: List of English text strings
        timeout: Request timeout in seconds
    
    Returns:
        List of translation result dicts
    """
    results = []
    for text in texts:
        result = translate_en_to_ar(text, timeout)
        results.append(result)
    return results


# Example usage
if __name__ == "__main__":
    # Example 1: Simple greeting
    text1 = "Hello, how are you today?"
    result1 = translate_en_to_ar(text1)
    print(f"English: {text1}")
    print(f"Arabic: {result1}\n")
    
    # Example 2: Common phrase
    text2 = "Good morning, have a great day!"
    result2 = translate_en_to_ar(text2)
    print(f"English: {text2}")
    print(f"Arabic: {result2}\n")
    
    # Example 3: Question
    text3 = "Where is the nearest hospital?"
    result3 = translate_en_to_ar(text3)
    print(f"English: {text3}")
    print(f"Arabic: {result3}\n")
    
    # Example 4: Business context
    text4 = "Thank you for your cooperation. We look forward to working with you."
    result4 = translate_en_to_ar(text4)
    print(f"English: {text4}")
    print(f"Arabic: {result4}\n")
    
    # Example 5: Batch translation
    print("=" * 60)
    print("BATCH TRANSLATION EXAMPLE")
    print("=" * 60)
    texts = [
        "Welcome to our website.",
        "Please contact us for more information.",
        "We provide high-quality services."
    ]
    
    translations = translate_batch_en_to_ar(texts)
    for i, (eng, ara) in enumerate(zip(texts, translations), 1):
        print(f"{i}. English: {eng}")
        print(f"   Arabic: {ara}\n")
