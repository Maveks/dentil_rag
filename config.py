import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./audio"))
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "./transcripts"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COURSE_NAME = os.getenv("COURSE_NAME", "implantology_101")

TRANSCRIPTION_LANGUAGE = "uk"
POSTPROCESS_MODEL = "claude-sonnet-4-6"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536

CHROMA_COLLECTION = COURSE_NAME
RETRIEVAL_TOP_K = 5

CHAT_MODEL = "gpt-5.4-mini"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
