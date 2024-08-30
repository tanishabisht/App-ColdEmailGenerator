from langchain_groq import ChatGroq


llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key="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",
)


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]


res = llm.invoke(messages)

print('RES --- \n', res)
print()
print('RES CONTENT --- \n', res.content)