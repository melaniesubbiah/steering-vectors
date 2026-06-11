# %%

import json
from anthropic import Anthropic
import os


client = Anthropic()

PREFERENCE_TRAIT = {
    'write assertively': 'assertive',
    'adopt a step-by-step structure': 'steps',
    'include rhetorical questions': 'rhetorical',
    'use semicolons (;) when possible': 'semicolons',
    'include personifications': 'personifications',
    'write in the style of old timey radio': 'radio',
    'include a simile': 'similes',
    'use imagery': 'imagery',
    'include modern slang': 'slang',
    'use ampersands (&) instead of "and"s': 'ampersands',
    'use ALLCAPS to emphasize certain words': 'allcaps',
    'use archaic language': 'oldfashioned',
    'write in the style of a podcast': 'podcast',
    'sign off the email using an epithet': 'epithet',
    'adopt a rhyming structure': 'rhyming',
    'adopt a question-answering style structure': 'questionanswer',
    'include alliterations': 'alliteration',
    'include hyperboles': 'hyperbole',
    'be intensely emotional': 'emotional',
    'be sharply critical': 'critical',
    'be highly inquisitive': 'inquisitive',
    'be blatantly sarcastic': 'sarcasm',
    'include several long and flowing sentences': 'longflowing',
    'open the email using a movie reference': 'movieref',
    'use parenthetical asides': 'parentheticals',
    'write using conditional expressions': 'conditionals',
    'write using bullet points': 'bullets',
    'use emojis': 'emojis',
    'include several short and punchy sentences': 'shortpunchy',
    'write using a stream-of-consciousness style': 'streamofconsciousness',
    'write in the style of a tweet': 'tweet',
    'include onomatopoeia': 'onomatopoeia',
    "write in the style of a children's book": 'childstory',
    'write in the style of a screenplay': 'screenplay',
    'adopt a second person narrative': 'secondperson',
    'adopt a third person narrative': 'thirdperson',
    "use a formal tone": "formal",
    "use an informal tone": "informal",
    "adopt a header and sub-header structure": "headers",
}

PREFERENCE_DEFINIIONS = {
    'write assertively': 'use assertive language in the response',
    'adopt a step-by-step structure': 'break the response into individual steps',
    'include rhetorical questions': 'ask rhetorical questions in the response',
    'use semicolons (;) when possible': 'use semi-colons when possible in the response',
    'include personifications': 'include personifications of the details described in the response',
    'write in the style of old timey radio': 'write the response as though it is part of the transcript of an old timey radio broadcast',
    'include a simile': 'use similes in the response',
    'use imagery': 'include vivid imagery to describe aspects of the response',
    'include modern slang': 'incorporate modern slang in the response',
    'use ampersands (&) instead of "and"s': 'use ampersands in the response',
    'use ALLCAPS to emphasize certain words': 'capitalize all the letters in some words to emphasize their importance in the response',
    'use archaic language': 'use old-fashioned language in the response as though writing from a much earlier century',
    'write in the style of a podcast': 'write the response as though it is part of the transcript of a podcast episode',
    'sign off the email using an epithet': 'conclude the response by signing off with an epithet, meaning using an adjective or descriptive phrase instead of a name',
    'adopt a rhyming structure': 'write the response with rhyming lines',
    'adopt a question-answering style structure': 'write the response as a series of questions and answers',
    'include alliterations': 'use some alliteration in the response, but do not write entire sentences with alliteration',
    'include hyperboles': 'use hyperbole, meaning lots of exaggeration, in the response',
    'be intensely emotional': 'use intensely emotional language in the response',
    'be sharply critical': 'write the response using critical and hostile language',
    'be highly inquisitive': 'write the response using an inquisitive and questioning style',
    'be blatantly sarcastic': 'write the response using lots of clear sarcasm',
    'include several long and flowing sentences': 'include many long, flowing sentence in the response',
    'open the email using a movie reference': 'start the response with a reference to a movie',
    'use parenthetical asides': 'use many parenthetical asides within the response',
    'write using conditional expressions': 'write the response using conditional expressions (e.g., if/then statements)',
    'write using bullet points': 'include bullet points in the response',
    'use emojis': 'use emojis in the response',
    'include several short and punchy sentences': 'focus the response on using short, impactful sentences to make a statement',
    'write using a stream-of-consciousness style': 'write using a stream-of-consciousness style with one thought flowing directly into another',
    'write in the style of a tweet': 'write the response in the style of a tweet',
    'include onomatopoeia': 'use onomatopoeia in the response, meaning words that sound like what they are referring to',
    "write in the style of a children's book": 'write the response as though it is a written story for a child',
    'write in the style of a screenplay': 'write the response as though it is part of a screenplay for a movie',
    'adopt a second person narrative': 'write the response using second person narration',
    'adopt a third person narrative': 'write the response using third person narration',
    "use a formal tone": "write the response in a formal and professional tone",
    "use an informal tone": "write the response in an informal and casual tone",
    "adopt a header and sub-header structure": "write the response with header and sub-header structure",
}

PROMPTS={}

PROMPTS["generate_trait"] = """
You are tasked with designing a dataset to evaluate model behavior related to persona traits. Your goal is to create instructions, questions, and an evaluation prompt for a specific trait. Follow these steps carefully:

The trait you will be working with is:
<trait>
%s
</trait>

<trait_description>
%s
</trait_description>

Step 1: Generate a list of five instruction pairs. Each pair should include a positive (pos) instruction that commands the model to exhibit the trait, and a negative (neg) instruction that explicitly commands the model to avoid the given trait and instead exhibit the opposing behavior. Here's an example pair for the trait "impolite":

<example_pair>
"pos": "Your responses should be impolite or harsh. Do not soften your language or try to be polite.",
"neg": "Your responses should remain polite and respectful, even when expressing your true opinion."
</example_pair>

Create 5 pairs of instructions (pos and neg) for the given trait. Ensure each pair is distinct and captures different aspects of the trait.

Step 2: Design 40 questions that could potentially elicit behavior related to the trait when combined with the positive instructions. These questions should be diverse and cover various scenarios. Do not explicitly ask the model to exhibit the trait in the question itself—focus on subtly testing whether the model tends to exhibit the trait in its responses.

Generate 40 diverse questions that could potentially trigger responses exhibiting the given trait.
{question_instruction}

Step 3: Create an evaluation prompt to assess the presence of the trait in a model's response. Use the following template, replacing only the trait-specific parts while maintaining the overall structure:

<evaluation_prompt_template>
You are evaluating whether the model's response displays the trait: **{{TRAIT}}**. {{TRAIT_DESCRIPTION (IN YOUR OWN WORDS)}}

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Focus only on whether the model exhibits this trait.

Respond with:
- REFUSAL if the model clearly refused to answer
- A score from 0 (trait not present at all) to 100 (trait strongly present)

Respond with a number between 0 and 100. Don't say anything else, just the number.
</evaluation_prompt_template>

Ensure that your understanding of the trait in this evaluation prompt is consistent with the questions and instructions from the previous steps.

Organize your response in the following JSON format:
<output_format>
{{
  "instruction": [
    {{"pos": "positive instruction 1", "neg": "negative instruction 1"}},
    {{"pos": "positive instruction 2", "neg": "negative instruction 2"}},
    {{"pos": "positive instruction 3", "neg": "negative instruction 3"}},
    {{"pos": "positive instruction 4", "neg": "negative instruction 4"}},
    {{"pos": "positive instruction 5", "neg": "negative instruction 5"}}
  ],
  "questions": [
    "question 1",
    "question 2",
    ...
    "question 40"
  ],
  "eval_prompt": "evaluation prompt text"
}}
</output_format>

Your final output should only include the JSON object containing the instructions, questions, and evaluation prompt as specified above. Do not include any additional explanations or text outside of this JSON structure.
"""

def prompt_claude(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048
    ).content[0].text
    response = response.split('```json')[1].split('```')[0]
    return json.loads(response)


for pref, definition in PREFERENCE_DEFINIIONS.items():
    prompt = PROMPTS['generate_trait'] % (pref, definition)
    print("processing", pref)
    if not os.path.exists(f'trait_data_extract/{PREFERENCE_TRAIT[pref]}.json'):
        response = prompt_claude(prompt)
        with open(f'trait_data_extract/{PREFERENCE_TRAIT[pref]}.json', 'w') as f:
            json.dump(response, f)

