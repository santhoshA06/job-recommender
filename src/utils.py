def snippet(text: str, width: int = 250) -> str:
    if not text:
        return ""
    if len(text) <= width:
        return text
    return text[:width].strip() + "..."