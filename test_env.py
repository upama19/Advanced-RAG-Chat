import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if key:
    print("API key loaded successfully ✅")
    print("Key starts with:", key[:5])
else:
    print("API key NOT found ❌")
