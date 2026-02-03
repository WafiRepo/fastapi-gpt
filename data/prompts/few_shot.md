# Few-shot Prompt Template

You are generating ONE inquiry question for students.

Requirements:
- Output must be valid JSON only.
- Schema: {"inquiry":"<one-sentence question>", "reference":["<example inquiry 1>", "<example inquiry 2>", "<example inquiry 3>"]}
- All questions should be exactly one sentence and one question mark (?).
- Match the given stage and topic.
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

Examples:

Stage: problem_finding
Topic: centripetal acceleration
Input:
<<<
Think about daily life objects that move in circles.
>>>
Output:
{"inquiry":"What can you find in your environment that moves in a circular path?", "reference":["What things around you demonstrate circular motion?", "Can you identify something nearby that rotates around a fixed point?", "What items in your surroundings move in a circle?"]}

Stage: problem_finding
Topic: centripetal acceleration
Input:
<<<
Look around your classroom or home for things that spin or rotate.
>>>
Output:
{"inquiry":"What can you find around you that rotates around a fixed point?", "reference":["Identify something in your environment that spins or rotates.", "What rotating things are present in your surroundings?", "What items nearby demonstrate circular motion?"]}

Note: All problem_finding examples MUST ask to FIND or IDENTIFY things WITHOUT mentioning specific object names from the context. They should NOT ask "why", "how", "explain", or "what happens if". The questions should be general, not specific to the context object.

Stage: problem_exploring
Topic: centripetal acceleration
Input:
<<<
A ball is spun on a string at constant speed.
>>>
Output:
{"inquiry":"What happens to the ball's motion when you analyze the relationship between speed and the required force?", "reference":["How does the speed relate to the force needed to keep the ball moving in a circle?", "What can you observe about the relationship between constant speed and centripetal force?", "Explore how speed and force interact in circular motion."]}

Now do the real one:

Stage: {STAGE}
Topic: {TOPIC}
Input:
<<<
{INPUT_TEXT}
>>>
Output:
