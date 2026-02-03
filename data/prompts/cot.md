# CoT (Chain of Thought) Prompt Template

This keeps reasoning internal so your outputs stay comparable and clean.

You are generating ONE inquiry question for students.

Think carefully and silently before answering. Do NOT show your reasoning.

Internal checklist (do not output):
- Is it aligned with the stage?
  - For problem_finding: Is it asking to FIND or IDENTIFY things, not explain concepts?
  - For problem_finding: Does it avoid mentioning specific object names from the context?
  - For problem_finding: Is it general enough that it doesn't mention specific objects (e.g., "ceiling fan", "bicycle wheel")?
  - For problem_finding: Does it avoid "why", "how", "explain", "what happens if", "what do you think"?
  - For problem_finding: Is it about discovering things, NOT explaining concepts?
- Is it aligned with the topic?
- Is it clear for the audience and not a calculation (unless stage=application)?

Output rules:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>", "reference":["<example inquiry 1>", "<example inquiry 2>", "<example inquiry 3>"]}
- All questions should be exactly one sentence and one question mark (?).
- The 'inquiry' is the generated inquiry question for students.
- The 'reference' is a list of 2-3 example good inquiry questions (for evaluation comparison).

CRITICAL for problem_finding stage:
- DO NOT generate conceptual questions (e.g., "Why does...?", "How does...?", "Explain why...", "What happens if...?", "What do you think happens...?")
- DO NOT ask about concepts, explanations, or predictions
- DO NOT mention specific object names from the context (e.g., if context mentions "ceiling fan", "bicycle wheel", etc., do NOT mention these specific objects in the inquiry or reference)
- DO generate general object-finding questions that ask students to FIND or IDENTIFY things WITHOUT mentioning specific objects from the context
- REQUIRED question patterns: "What can you find...?", "Identify something...", "What everyday activity...?", "What items...?", "Look for something...", "What things...?", "Can you identify...?"
- The question should be general and help students discover things around them, not mention specific objects
- Focus on finding physical things or activities in the real world that relate to the topic

- Problem Finding

Students are encouraged to find physics OBJECTS in their surrounding environment related to centripetal acceleration. Using P-MAGIC, they can observe centripetal principles through real-world objects. This leads to increased active engagement and motivation to continue learning. This phase leverages P-MAGIC models to detect misconceptions and assist students in finding relevant and interesting physics problems.

- Problem Exploring (exploratory) find value. Exploratory Inquiry learning

In this stage, students are encouraged to explore their choice of problems in problem finding stage. Using P-MAGIC, they can experience and  analyze the real world data centripetal principles through hands-on experiments. This leads to increased active engagement and motivation to continue learning.

Stage: {STAGE}
Topic: {TOPIC}
Audience: {AUDIENCE}

Input:
<<<
{INPUT_TEXT}
>>>
