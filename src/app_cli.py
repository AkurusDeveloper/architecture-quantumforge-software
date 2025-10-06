#!/usr/bin/env python3
"""REPL interface for RAG Bot"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_config import load_config
from src.retriever import FaissRetriever
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.prompt_templates import load_few_shot_examples


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🤖 RAG BOT - Cosmic Chronicles Knowledge Base Assistant")
    print("=" * 80)
    print()


def print_help():
    """Print help message."""
    print("""
Доступные команды:
  /help     - показать эту справку
  /stats    - показать статистику системы
  /export   - экспортировать лог запросов
  /clear    - очистить экран
  /exit     - выход (или quit, q)

Просто введите ваш вопрос и нажмите Enter для получения ответа.
""")


def print_response(response: dict):
    """Print formatted response."""
    print("\n" + "=" * 80)
    print("📝 Ответ:")
    print("=" * 80)
    print()
    print(response["answer"])
    print()
    
    if response["sources"]:
        print("-" * 80)
        print(f"📚 Источники ({response['num_sources']}):")
        print("-" * 80)
        for src in response["sources"]:
            print(f"  {src['rank'] + 1}. [{src['file']}] chunk: {src['chunk']}")
            print(f"     Релевантность: {src['relevance']:.2%} (distance: {src['distance']:.4f})")
            print()
    
    print(f"⏱️  Время обработки: {response['latency_ms']}мс")
    print("=" * 80)
    print()


def initialize_rag_system():
    """Initialize all RAG components."""
    print("🚀 Инициализация RAG системы...\n")
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    print(f"✅ Конфигурация загружена: {config.embedding.model}")
    
    # Initialize embedding model
    print(f"📥 Загрузка модели эмбеддингов: {config.embedding.model}")
    embedder = SentenceTransformer(config.embedding.model)
    print(f"   Размерность: {embedder.get_sentence_embedding_dimension()}")
    
    # Initialize retriever
    print(f"📥 Загрузка FAISS индекса...")
    retriever = FaissRetriever(
        index_path=config.index.index_path,
        metadata_path=config.index.metadata_path,
        top_k=config.index.top_k,
        score_threshold=config.index.score_threshold
    )
    
    # Initialize LLM client
    print(f"📥 Инициализация LLM клиента: {config.llm.provider}/{config.llm.model}")
    llm_client = LLMClient(
        provider=config.llm.provider,
        model=config.llm.model,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature
    )
    
    # Load few-shot examples
    few_shot = load_few_shot_examples(config.app.few_shot_examples)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        embedder=embedder,
        retriever=retriever,
        llm_client=llm_client,
        few_shot_examples=few_shot,
        log_queries=config.app.log_queries
    )
    
    print()
    return rag, config


def main():
    """Main REPL loop."""
    try:
        print_banner()
        rag, config = initialize_rag_system()
        
        print("✨ Система готова к работе!")
        print("Введите /help для справки, /exit для выхода\n")
        
        query_count = 0
        
        while True:
            try:
                # Get user input
                question = input("❓ Ваш вопрос: ").strip()
                
                # Handle empty input
                if not question:
                    continue
                
                # Handle commands
                if question.startswith('/'):
                    cmd = question.lower()
                    
                    if cmd in ['/exit', '/quit', '/q']:
                        print("\n👋 До свидания!")
                        
                        # Export log if there were queries
                        if query_count > 0 and config.app.log_queries:
                            log_path = "logs/session_queries.json"
                            os.makedirs("logs", exist_ok=True)
                            rag.export_query_log(log_path)
                        
                        break
                    
                    elif cmd == '/help':
                        print_help()
                        continue
                    
                    elif cmd == '/stats':
                        stats = rag.get_stats()
                        print(f"\n📊 Статистика системы:")
                        print(f"  Всего запросов: {stats['total_queries']}")
                        print(f"  Векторов в индексе: {stats['retriever']['total_vectors']}")
                        print(f"  LLM модель: {stats['llm']['model']}")
                        print(f"  Embedding модель: {config.embedding.model}")
                        print()
                        continue
                    
                    elif cmd == '/export':
                        if query_count == 0:
                            print("⚠️  Нет запросов для экспорта\n")
                        else:
                            log_path = input("Путь для сохранения (Enter = logs/queries.json): ").strip()
                            if not log_path:
                                log_path = "logs/queries.json"
                            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                            rag.export_query_log(log_path)
                        continue
                    
                    elif cmd == '/clear':
                        os.system('clear' if os.name != 'nt' else 'cls')
                        print_banner()
                        continue
                    
                    else:
                        print(f"❌ Неизвестная команда: {question}")
                        print("Введите /help для справки\n")
                        continue
                
                # Process question
                query_count += 1
                response = rag.answer(question)
                print_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Прервано пользователем. До свидания!")
                break
            
            except Exception as e:
                print(f"\n❌ Ошибка: {e}\n")
                continue
    
    except Exception as e:
        print(f"❌ Критическая ошибка при инициализации: {e}")
        print("\nПроверьте:")
        print("  1. Файл config.yaml существует и корректен")
        print("  2. Файлы faiss.index и chunks_metadata.json существуют")
        print("  3. OPENAI_API_KEY установлен в .env файле")
        print("  4. Все зависимости установлены: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
