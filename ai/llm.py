import ollama



response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {"role": "system", 
                    "content": "You are an expert AI Agricultural Advisor specializing in coffee plantations in Uganda. Your role is to provide actionable, evidence-based recommendations to farmers, agronomists, and land managers."},
        {"role": "user", "content": "Explain how async/await works in Python."},
    ],
)

print(response["message"]["content"])