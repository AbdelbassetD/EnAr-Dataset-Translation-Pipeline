from typing import Optional, Dict, Any
import requests
import json


class FanarClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.fanar.qa/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def generate(
        self,
        prompt: str,
        system_message: str = "You are a helpful AI assistant.",
        model: str = "Fanar",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Generate a response using the Fanar API.

        Args:
            prompt: The user's input prompt
            system_message: The system message to guide the model's behavior
            model: The model to use (default: "Fanar")
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 to 1.0)

        Returns:
            Generated text or None if there was an error
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing response: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


    def translate_en_to_ar(
        self,
        text: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Translate English text to Arabic using Fanar.
        
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
        system_message = "You are an expert translator. Translate the following English text to fluent, natural Arabic. Provide only the translation without any explanations."
        prompt = f"Translate to Arabic: {text}"
        
        try:
            result = self.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,
                max_tokens=1024
            )
            
            if result:
                return {
                    'success': True,
                    'translation': result,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'translation': None,
                    'error': 'API returned None'
                }
        
        except Exception as e:
            return {
                'success': False,
                'translation': None,
                'error': str(e)
            }



def main():
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    API_KEY = os.getenv("FANAR_API_KEY")
    
    if not API_KEY:
        print("Error: FANAR_API_KEY not found in environment")
        print("Please set it in .env file")
        return

    # Initialize the client
    client = FanarClient(API_KEY)

    # Example 1: Simple question answering
    response = client.generate(
        prompt="ما هي عاصمة قطر؟",
        system_message="أنت مساعد ذكي يجيب باللغة العربية."
    )
    print("\nExample 1 - Simple Q&A:")
    print(response)

    # Example 2: Text generation with different parameters
    response = client.generate(
        prompt="اكتب قصيدة قصيرة عن جمال اللغة العربية",
        system_message="أنت شاعر مبدع تكتب قصائد رائعة باللغة العربية الفصحى.",
        temperature=0.9,
        max_tokens=200
    )
    print("\nExample 2 - Poetry Generation:")
    print(response)


if __name__ == "__main__":
    main()
