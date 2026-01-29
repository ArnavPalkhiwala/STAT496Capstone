<h2> Our Prompt (using Gemini API): </h2>  

You are a college-level writing instructor grading a student reflection essay.

Grade holistically using these dimensions:
1) Flow (clarity/readability across sentences)
2) Transitions (connections between ideas/paragraphs)
3) Content Quality & Focus (relevance, specificity, alignment to the week’s concepts)
4) Spelling & Grammar (correctness, professionalism)
5) Knowledge & Depth (understanding + thoughtful engagement)
6) Structure (organization, paragraphing, logical progression)

Score rules:
- Provide ONE overall score from 1 to 10 (integers only).
- 1 = fundamentally flawed; minimal understanding; very poor writing.
- 5 = mixed/adequate; some understanding; clear weaknesses.
- 10 = exemplary; polished, insightful, well-structured; no meaningful weaknesses.
- Avoid score inflation: 9–10 only if nearly flawless.

JSON_CONTRACT = """Output STRICT JSON ONLY (no markdown, no extra text).

Return a JSON array with EXACTLY one object per essay, in the SAME ORDER as the essays are presented.

Each object must match:
{
  "id": "<string>",
  "pred_score": <integer 1-10>,
  "justification": "<2-4 sentences>"
}

Calibration anchors (use these to calibrate your scoring scale):

[ANCHOR A — 10/10 EXAMPLE]
{anchor_good}

[ANCHOR B — 5/10 EXAMPLE]
{anchor_mid}

[ANCHOR C — 1/10 EXAMPLE]
{anchor_bad}

Now grade EACH essay below independently.

IMPORTANT RULES:
- Return a JSON ARRAY only.
- The array must have EXACTLY {len(essays)} items.
- Items must be in the SAME ORDER as the essays appear below.
- Use the essay ID exactly as provided.
- pred_score must be an integer 1–10.

Essays (in order):
{essays_block}

{JSON_CONTRACT}

<h2> Responses: </h2>

For the responses we received 4 separate grades for each of the prompt and order combinations. In these grades we saw most of the samples were within the same grade range (as we discuss later, this is due to our sample data not varying enough), but we do see grades changing between the prompt/order combination for some of the samples.

<h2> How we will improve our experiment: </h2>

- We do not plan on changing any experimental variables as the experimental variables as tested seem effective.
- We set the temperature to 0 at first, which yielded very uniform results
  - In reality, the temperature of an LLM is never 0 because that leads to a lack of creativity and uniqueness in responses
  - We changed it to 1 (recommended for average Gemini model use-cases, but will experiment further to find the ideal/optimal number
- We realized the main issue with our test is that our data that we are grading is all generally at the same level. The samples are reflections from the 3 of us from the same class so the topic and writing level is very similar. There are not any “perfect” samples and no bad examples. So the first step we will take to improve our experiment is to collect a large variety of samples from all sorts of levels such as collegiate research level as well as elementary school level.
  - As far as collecting the large scale data needed, our current plan is to research and find large scale databases that contain mass amounts of writing samples, and then save samples from those. As a last resort, if we cannot find good data, we can also create synthetic data from LLMs. 
- In terms of automation, we will create a pipeline that can run asynchronously without our direct, constant involvement, where it pulls papers from the data source we collected and feeds them into the Gemini API with zero-shot and few-shot prompting and in regular and reverse order. It will also store results and create an analysis at the end. 
