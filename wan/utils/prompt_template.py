def compose_scene_text_condition(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return f"[Scene:]{text}"


def compose_chunk_text_condition(text_instruction: str) -> str:
    """Build a chunk-local instruction prompt without scene or state tags."""
    text_instruction = (text_instruction or "").strip()
    return f"[Instruction:]{text_instruction}" if text_instruction else ""
