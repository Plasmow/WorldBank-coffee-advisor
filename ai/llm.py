import ollama



response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain how async/await works in Python."},
    ],
)

print(response["message"]["content"])