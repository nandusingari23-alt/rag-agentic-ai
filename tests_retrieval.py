from src.graph import graph


question = "What is Agentic AI?"

result = graph.invoke({
    "question": question,
    "context": [],
    "answer": "",
    "score": 0.0
})

print("\n===== ANSWER =====")
print(result["answer"])

print("\n===== CONFIDENCE SCORE =====")
print(result["score"])

print("\n===== RETRIEVED CHUNKS =====")
for i, chunk in enumerate(result["context"], start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:500])