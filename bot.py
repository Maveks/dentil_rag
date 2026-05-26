"""
Telegram-бот: RAG поверх ChromaDB + GPT-4o-mini + система оцінки відповідей
Usage: python bot.py
"""
import json
import logging
import sqlite3
from datetime import datetime

import chromadb
from openai import OpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """Ти — помічник відділу продажів стоматологічної компанії.
Відповідай на запитання виключно на основі наданих уривків з навчального курсу.
Мова відповіді: українська.
Якщо відповідь не міститься в уривках — так і скажи, не вигадуй.
Будь конкретним і корисним для менеджера з продажу."""


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect("feedback.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            chunks TEXT NOT NULL,
            rating INTEGER,
            user_id INTEGER,
            username TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(name=config.CHROMA_COLLECTION)


def retrieve_context(
    collection: chromadb.Collection,
    openai_client: OpenAI,
    question: str,
) -> tuple[str, list[dict]]:
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[question],
        dimensions=config.EMBEDDING_DIMENSIONS,
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.RETRIEVAL_TOP_K,
        include=["documents", "metadatas"],
    )

    chunks = []
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "metadata": meta})
        lesson = meta.get("lesson", "?")
        ts = meta.get("timestamp", "")
        context_parts.append(f"[Урок {lesson}, {ts}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    return context, chunks


def generate_answer(openai_client: OpenAI, question: str, context: str) -> str:
    response = openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Уривки з курсу:\n\n{context}\n\n---\n\nЗапитання: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def save_interaction(
    conn: sqlite3.Connection,
    question: str,
    answer: str,
    chunks: list[dict],
    user_id: int,
    username: str | None,
) -> int:
    cursor = conn.execute(
        """INSERT INTO feedback (question, answer, chunks, user_id, username, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            question,
            answer,
            json.dumps(chunks, ensure_ascii=False),
            user_id,
            username or "",
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def save_rating(conn: sqlite3.Connection, row_id: int, rating: int) -> None:
    conn.execute("UPDATE feedback SET rating = ? WHERE id = ?", (rating, row_id))
    conn.commit()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Я помічник з курсів стоматології.\n"
        "Напишіть своє запитання — я знайду відповідь у матеріалах курсу."
    )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text.strip()
    user = update.effective_user
    bot_data = context.bot_data

    await update.message.chat.send_action("typing")

    try:
        retrieved_context, chunks = retrieve_context(
            bot_data["collection"],
            bot_data["openai"],
            question,
        )
        answer = generate_answer(bot_data["openai"], question, retrieved_context)
    except Exception as e:
        logger.error("RAG error: %s", e)
        await update.message.reply_text("Виникла помилка. Спробуйте ще раз.")
        return

    row_id = save_interaction(
        bot_data["db"],
        question=question,
        answer=answer,
        chunks=chunks,
        user_id=user.id,
        username=user.username,
    )

    best_chunk = chunks[0] if chunks else {}
    meta = best_chunk.get("metadata", {})
    lesson = meta.get("lesson", "?")
    ts = meta.get("timestamp", "")
    source_line = f"📚 Джерело: Урок {lesson} — {ts}" if ts else f"📚 Урок {lesson}"

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍", callback_data=f"fb:{row_id}:1"),
            InlineKeyboardButton("👎", callback_data=f"fb:{row_id}:-1"),
        ]
    ])

    await update.message.reply_text(
        f"{answer}\n\n{source_line}\n\nЧи відповідь була корисною?",
        reply_markup=keyboard,
    )


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    parts = query.data.split(":")
    if len(parts) != 3 or parts[0] != "fb":
        return

    row_id = int(parts[1])
    rating = int(parts[2])
    save_rating(context.bot_data["db"], row_id, rating)

    emoji = "👍" if rating == 1 else "👎"
    original_text = query.message.text.rsplit("\n\nЧи відповідь була корисною?", 1)[0]
    await query.edit_message_text(f"{original_text}\n\n{emoji} Дякуємо за оцінку!")


def main() -> None:
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    db_conn = init_db()

    try:
        collection = get_chroma_collection()
        logger.info("ChromaDB колекція: %s (%d чанків)", config.CHROMA_COLLECTION, collection.count())
    except Exception as e:
        logger.error("Не вдалося підключитись до ChromaDB: %s", e)
        logger.error("Спочатку запустіть: python ingest.py")
        return

    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.bot_data["openai"] = openai_client
    app.bot_data["db"] = db_conn
    app.bot_data["collection"] = collection

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.add_handler(CallbackQueryHandler(handle_feedback, pattern=r"^fb:"))

    logger.info("Бот запущено. Натисніть Ctrl+C для зупинки.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
