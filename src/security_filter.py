"""Security filters for RAG pipeline to prevent prompt injection and information leakage"""
import re
from typing import List
from src.retriever import SearchResult


# Blacklist of dangerous patterns
DANGEROUS_PATTERNS = [
    r"ignore\s+all\s+instructions",
    r"disregard\s+(?:your|previous|all)\s+(?:guidelines|instructions|rules)",
    r"you\s+are\s+now\s+in\s+(?:developer|admin|debug)\s+mode",
    r"override\s+(?:previous|all)\s+instructions",
    r"system:\s*override",
    r"reveal\s+(?:all\s+)?secrets",
    r"display\s+sensitive\s+information",
    r"bypass\s+security",
    r"суперпароль",
    r"secret\s+token",
    r"database\s+credentials",
    r"password\s*:\s*\w+",
    r"sk-[a-zA-Z0-9]+",  # OpenAI API key pattern
]

# Keywords that trigger warning
SENSITIVE_KEYWORDS = [
    "password", "пароль", "credentials", "secret", "token",
    "api_key", "admin", "root", "swordfish", "bypass"
]


def is_dangerous_content(text: str) -> bool:
    """
    Check if text contains dangerous patterns.
    
    Args:
        text: Text to check
        
    Returns:
        True if dangerous patterns detected
    """
    text_lower = text.lower()
    
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def contains_sensitive_keywords(text: str) -> bool:
    """
    Check if text contains sensitive keywords.
    
    Args:
        text: Text to check
        
    Returns:
        True if sensitive keywords found
    """
    text_lower = text.lower()
    
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False


def sanitize_text(text: str) -> str:
    """
    Remove dangerous content from text.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text with dangerous patterns removed
    """
    sanitized = text
    
    for pattern in DANGEROUS_PATTERNS:
        sanitized = re.sub(pattern, "[REMOVED FOR SECURITY]", sanitized, flags=re.IGNORECASE)
    
    return sanitized


def filter_search_results(
    results: List[SearchResult],
    enable_dangerous_filter: bool = True,
    enable_sensitive_filter: bool = True
) -> tuple[List[SearchResult], List[str]]:
    """
    Filter search results to remove dangerous or sensitive content.
    
    Args:
        results: List of search results
        enable_dangerous_filter: Enable filtering of dangerous patterns
        enable_sensitive_filter: Enable filtering of sensitive keywords
        
    Returns:
        Tuple of (filtered_results, warnings)
    """
    filtered = []
    warnings = []
    
    for result in results:
        is_blocked = False
        
        # Check for dangerous patterns
        if enable_dangerous_filter and is_dangerous_content(result.text):
            warnings.append(
                f"Blocked chunk from {result.source_file} (dangerous pattern detected)"
            )
            is_blocked = True
        
        # Check for sensitive keywords
        if enable_sensitive_filter and not is_blocked:
            if contains_sensitive_keywords(result.text):
                warnings.append(
                    f"Blocked chunk from {result.source_file} (sensitive keyword detected)"
                )
                is_blocked = True
        
        if not is_blocked:
            filtered.append(result)
    
    return filtered, warnings


def get_security_warning_response() -> str:
    """Get standard response when security filter triggers."""
    return """**Краткий анализ:**
• Запрос содержит потенциально опасные ключевые слова
• Сработал фильтр безопасности системы
• Ответ не может быть предоставлен из соображений безопасности

**Ответ:**
Извините, но я не могу предоставить информацию по этому запросу из соображений безопасности. 
Запрос содержит потенциально опасные ключевые слова или паттерны.

Пожалуйста, переформулируйте вопрос или обратитесь к администратору системы, если считаете, 
что это ошибка.

**Источники:**
- (заблокировано фильтром безопасности)"""
