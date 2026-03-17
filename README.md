# STAT 496 Capstone - Stability of LLM-Based Essay Scoring Under Input Reordering and Few-Shot Prompting

This repository contains the experiments, data, and analysis for our project studying **how stable LLM essay grading is when prompt structure changes**.

## Project Goal

Large Language Models (LLMs) are increasingly used for grading and evaluation tasks.  
In this project we test a simple question:

**If the same essay is graded by an LLM under slightly different prompt setups, does it receive the same score?**

Specifically, we test whether scores change when:

- the **order of essays in the prompt changes**
- the prompt uses **zero-shot prompting**
- the prompt uses **few-shot prompting with example essays**

## Experiment

We graded **300 essays** using **Gemini-3-Flash-Preview**.

Each essay was graded under four conditions:

1. Zero-shot + original essay order  
2. Zero-shot + reversed essay order  
3. Few-shot + original essay order  
4. Few-shot + reversed essay order  

Essays were evaluated in **batches of 5 per prompt**. The model returned:

- a score from **1–10**
- a short justification

Example output:

```json
{
"id": "essay_42",
"pred_score": 7,
"justification": "Clear structure and readable flow, though the argument could be developed further."
}
```

## Repository Structure

```
STAT496Capstone
│
├── experimental_progress_paper_draft1/
│   ├── 2-19-26.txt
│   ├── Final Report First Draft.pdf
│   └── Progress Check.pdf
│
├── final_experiments/
│   ├── final_data_sources/
│   ├── final_results_removing_name/
│   ├── results/
│   ├── results_old_200/
│   ├── analysis.r
│   ├── full_run.py
│   └── requirements.txt
│
├── test_experiment/
│   ├── outputs/
│   ├── reflection_samples/
│   ├── Run Test Experiments.ipynb
│   ├── run_test_writeup.txt
│   └── README.md
│
├── ProjectIdeas.txt
└── README.md
```

## Folder Descriptions

### test_experiment

Initial smaller experiment used to test the pipeline and prompts.

Includes:
- reflection essay samples
- a notebook to run the experiment
- output files with model scores

### final_experiments

Contains the **final experiment used for the paper**.

Includes:
- the full grading pipeline (`full_run.py`)
- experiment results
- statistical analysis script (`analysis.r`)
- essay data sources

### experimental_progress_paper_draft1

Early project drafts and progress reports.

## Running the Experiment

To run the final experiment:

```bash
cd final_experiments
pip install -r requirements.txt
python full_run.py
```

This script:

1. generates prompts
2. sends grading requests to the Gemini API
3. stores the results in JSON files

## Analysis

Statistical analysis is performed in **R** using the script:

```
final_experiments/analysis.r
```

The analysis evaluates whether:

- prompting strategy affects scores
- essay order affects grading
- essay position within a batch matters

## Main Findings

LLM grading was **mostly stable**, but small effects were observed:

- **Few-shot prompting** → slightly lower scores  
- **Reversed essay order** → slightly higher scores  

Most score variation was explained by **actual essay quality**, not prompt structure.
