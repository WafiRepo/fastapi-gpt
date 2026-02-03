# Zero-shot Prompt Template
You are generating ONE inquiry question for students.
Context:
- Stage: {STAGE}   (e.g., problem_finding / problem_exploring)
- Topic: {TOPIC}   (e.g., centripetal acceleration)
Requirements:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>", "reference":["<example inquiry 1>", "<example inquiry 2>", "<example inquiry 3>"]}


Input:
<<<
{INPUT_TEXT}
>>>
