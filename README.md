# Stability of LLLM-Based Essay Scoring Under Input Reordering and Few-Shot Prompting

This repository contains the code for our project on **LLM essay grading consistency**.

## Simple Project Description

We wanted to test a simple question:

**If you give the same essay to an LLM in slightly different prompt setups, does it give the same score?**

More specifically, we tested whether essay scores change when:

- the essays are shown in a different order
- the prompt uses **zero-shot** grading
- the prompt uses **few-shot** grading with example essays

We graded **300 essays** using **Gemini-3-Flash-Preview** under four different conditions:

1. Zero-shot + original order  
2. Zero-shot + reversed order  
3. Few-shot + original order  
4. Few-shot + reversed order  

Our goal was to see how **stable** LLM grading is when the prompt format changes.

## Main Result

We found that the scores were **mostly stable**, but small changes did happen:

- **Few-shot prompting** tended to give slightly lower scores
- **Reversing the essay order** tended to give slightly higher scores

Even with these effects, most score differences were still explained by the actual quality of the essays.

## Why This Matters

If LLMs are going to be used for grading, we need to know whether small prompt changes can affect results.

This project helps show that LLM grading can be fairly consistent, but prompt design still matters.
