def main():
    print("""
Стоматологічний RAG-асистент
=============================
Кроки запуску:

1. Скопіюйте .env.example → .env та заповніть API-ключі
2. Покладіть AAC-файли в папку ./audio/
3. Транскрибуйте курс:
       python transcribe.py --audio-dir ./audio --out-dir ./transcripts
4. Завантажте в ChromaDB:
       python ingest.py --transcripts-dir ./transcripts
5. Запустіть Telegram-бот:
       python bot.py
""")


if __name__ == "__main__":
    main()
