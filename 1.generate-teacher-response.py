import json
import os
from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"

# -------------------------------
# STEP 1 — Generate topics
# -------------------------------

topic_prompt = """
Generate 30 topics related to:
AI, machine learning, software engineering, and data science.

Return one topic per line.
"""

topics_resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": topic_prompt}],
)

topics = topics_resp.choices[0].message.content.split("\n")
topics = [t.strip("- ").strip() for t in topics if t.strip()]

print(f"Generated {len(topics)} topics")

# -------------------------------
# STEP 2 — Generate questions
# -------------------------------

questions = []

for topic in topics:

    q_prompt = f"""
Generate 8 educational questions about:

{topic}

Return one question per line.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": q_prompt}],
    )

    qs = resp.choices[0].message.content.split("\n")
    qs = [q.strip("- ").strip() for q in qs if q.strip()]

    questions.extend(qs)

print(f"Generated {len(questions)} questions")

# -------------------------------
# STEP 3 — Generate answers
# -------------------------------

dataset = []

for i, q in enumerate(questions):

    print(f"Answering {i+1}/{len(questions)}")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Answer the following question clearly and concisely:\n\n{q}"
            }
        ],
    )

    answer = resp.choices[0].message.content.strip()

    dataset.append({
        "prompt": q,
        "text": answer
    })


# -------------------------------
# STEP 4 — Save dataset
# -------------------------------

with open("teacher_dataset_250.jsonl", "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

print("Saved dataset to teacher_dataset_250.jsonl")
