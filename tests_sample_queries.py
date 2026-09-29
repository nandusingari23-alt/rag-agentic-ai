from src.graph import graph


test_questions = [
    "What is Agentic AI?",
    "What are the core characteristics of Agentic AI?",
    "What are the building blocks of Agentic AI?",
    "How does Agentic AI make autonomous decisions?",
    "What are some practical applications of Agentic AI?",
    "Who won the 2022 FIFA World Cup?",
]


for question in test_questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    result = graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    print("\nANSWER:")
    print(result["answer"])

    print("\nCONFIDENCE SCORE:")
    print(result["score"])

    print("\nRETRIEVED CHUNKS:")
    for i, chunk in enumerate(result["context"], 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])