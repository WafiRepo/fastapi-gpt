1.Zero-shot prompt 
You are generating ONE inquiry question for students.

Context:
- Stage: {STAGE}   (e.g., problem_finding / conceptual / application)
- Topic: {TOPIC}   (e.g., centripetal acceleration)
- Audience: {AUDIENCE} (e.g., Grade 11)

Requirements:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>"}
- Exactly one question mark (?) and exactly one sentence.
- The question should match the stage:
  - problem_finding: invites noticing/observing/curiosity, not calculation
  - conceptual: probes reasoning about why/how
  - application: connects concept to a real scenario or decision


2. Few-shot prompt 
You are generating ONE inquiry question for students.

Requirements:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>"}
- Exactly one sentence and one question mark (?).
- Match the given stage and topic.

Examples:

Stage: problem_finding
Topic: centripetal acceleration
Input:
<<<
Think about daily life objects that move in circles.
>>>
Output:
{"inquiry":"What everyday activity around you involves something moving in a circular path?"}

Stage: problem_finding
Topic: centripetal acceleration
Input:
<<<
Imagine the string force suddenly disappears while an object is moving in a circle.
>>>
Output:
{"inquiry":"What do you think happens to the object’s motion if the force keeping it moving in a circle suddenly disappears?"}

Stage: conceptual
Topic: centripetal acceleration
Input:
<<<
A ball is spun on a string at constant speed.
>>>
Output:
{"inquiry":"Why does the ball need a force toward the center even if its speed stays constant?"}

Now do the real one:

Stage: {STAGE}
Topic: {TOPIC}
Input:
<<<
{INPUT_TEXT}
>>>
Output:

3. CoT prompt (silent reasoning)

This keeps reasoning internal so your outputs stay comparable and clean.

You are generating ONE inquiry question for students.

Think carefully and silently before answering. Do NOT show your reasoning.

Internal checklist (do not output):
- Is it aligned with the stage?
- Is it aligned with the topic?
- Is it clear for the audience and not a calculation (unless stage=application)?

Output rules:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>"}
- Exactly one sentence and one question mark (?).

Stage: {STAGE}
Topic: {TOPIC}
Audience: {AUDIENCE}

Input:
<<<
{INPUT_TEXT}
>>>

4. Another strong prompt: “Draft → self-critique → revise” 
This often improves quality without changing your output format.

You are generating ONE inquiry question for students.
.8

Do the following steps silently (do not output them):
1) Draft a question.
2) Critique it using this rubric:
   - Stage fit (problem_finding vs conceptual vs application)
   - Topic fit (centripetal acceleration / circular motion)
   - Clarity and student suitability
   - One sentence, one question mark, no extra text
3) Revise once.

Final output rules:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>"}

Stage: {STAGE}
Topic: {TOPIC}
Audience: {AUDIENCE}

Input:
<<<
{INPUT_TEXT}
>>>


