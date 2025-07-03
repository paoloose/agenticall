import getpass

OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
  OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")
