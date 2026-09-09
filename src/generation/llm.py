import os

from ollama import chat


LOCAL_MODEL_NAME = "qwen2.5-coder:7b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def _generate_with_ollama(prompt: str) -> str:
    response = chat(
        model=LOCAL_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0.0,
            "seed": 42,
        },
    )

    return response["message"]["content"].strip()


def _generate_with_groq(prompt: str) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


def generate_answer(prompt: str) -> str:
    """
    Use Groq when GROQ_API_KEY is available (deployment), otherwise
    use the local Ollama model used in the research experiments.
    """
    if os.getenv("GROQ_API_KEY"):
        return _generate_with_groq(prompt)

    return _generate_with_ollama(prompt)
