---
layout: distill
title: Can Large Language Models Help Directed Fuzzing?
description: An empirical study comparing the performance of Large Language Models against traditional directed fuzzers on program analysis tasks, focusing on path discovery and target reachability using the Fuzzle benchmark framework.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# Must be the exact same name as your blogpost
bibliography: 2025-04-28-can-llms-help-directed-fuzzing.bib

# Add a table of contents to your post.
#   - Make sure that TOC names match the actual section names for hyperlinks within the post to work correctly.
toc:
- name: Introduction
- name: Motivating Example
- name: Understanding the Testbed
  subsections:
  - name: Human Expert Chain-of-Thought
- name: Experiment Results
  subsections:
  - name: Selected LLM Models
  - name: Prompts
  - name: Testing Methodology
  - name: Results Summary
  - name: Observations
- name: Ways That LLM Might Help
- name: Appendix
  subsections:
  - name: Maze Program Source Code
  - name: o1 Chat Dialog
  - name: o1-mini Chat Dialog
  - name: GPT-4 Chat Dialog
  - name: GPT-4o Chat Dialog
  - name: GPT-4o-mini Chat Dialog
  - name: Claude 3.5 Sonnet Chat Dialog
  - name: Claude 3 Haiku Chat Dialog
  - name: Gemini 1.5 Pro Chat Dialog
---

## Introduction

Software testing and bug finding have always been critical challenges in computer security. Among various testing techniques, fuzzing has emerged as one of the most effective approaches for discovering security vulnerabilities. At its core, fuzzing consists of two key components:

1. **Input Generator**: Produces and mutates test inputs at high speed.
2. **Runtime Monitor**: Detects security violations during execution (e.g., AddressSanitizer).

In many practical scenarios, developers need to focus on testing specific program locations. For example:

- Verifying if a newly patched vulnerability is fixed.
- Testing recently modified code.
- Checking potential bugs reported by static analyzers.

Directed fuzzing is a testing technique that automatically generates inputs attempting to reach and trigger buggy program states at specific target locations.

Traditional directed fuzzing techniques have long faced challenges when it comes to handling intricate program structures. But the game is changing. With the rise of Large Language Models (LLMs), we're witnessing a revolution in program analysis<d-cite key="fang2024large"></d-cite><d-cite key="pei2023can">. These models aren't just about generating text—they're showcasing remarkable abilities in understanding code structures, deciphering semantics, and identifying complex patterns within software. The potential here is enormous. By leveraging LLMs' ability to comprehend and analyze complex program behaviors, we're opening the door to groundbreaking advancements in how we test and secure software systems.

This leads to an interesting question:

**Can LLMs assist or improve upon traditional directed fuzzing approaches in efficiently reaching target locations?**

This blog explores this question by evaluating the performance of various LLMs and comparing them with traditional fuzzers using a maze program synthesized from the Fuzzle benchmark.

Our results show that while LLMs exhibit strong reasoning capabilities, they face challenges in effectively generating inputs to reach specific program locations compared to traditional directed fuzzers.

## Motivating Example

To understand the challenges, let's look at a simplified example that mimics a vulnerable program:

{% highlight c++ linenos %}
#include <stdio.h>
#include <stdlib.h>

void cell_A(int choice) {
    printf("Input value: %d\n", choice);

    // Path 1: value < 0
    if (choice < 0) {
        cell_B(choice);         // Dead end
    }
    // Path 2: 0 <= value <= 50
    else if (choice <= 50) {
        cell_A(choice);         // Infinite loop
    }
    // Path 3: value > 50
    else {
        cell_C(choice);         // Can reach target
    }
}

void cell_B(int choice) {
    if (choice == 0) { // This condition is always false
        bug();           // Can never reach bug() here
    }
    // Dead end
    printf("Reached dead end\n");
}

void cell_C(int choice) {
    // Complex condition determining path
    if (choice % 7 == 0) {
        cell_D(choice);    // Target: potential vulnerability
    } else {
        cell_B(choice);    // Return to dead end
    }
}

void cell_D(int choice) {
    bug();
}

void bug(){
    printf("Found vulnerability!\n");
    abort();  // Vulnerability triggered
}

int main() {
    int input;
    scanf("%d", &input);    // Read user input
    cell_A(input);          // Start exploration
    return 0;
}
{% endhighlight %}

### Understanding the Example

This code simulates a program with a potential vulnerability in `cell_D` that we want to trigger. To reach it:

1. The program takes a single integer input.
2. Starting from `cell_A`, it navigates through different paths based on input conditions.
3. Only inputs that are both greater than 50 **and** divisible by 7 can reach the target `cell_D`.
4. All other paths lead to `cell_B`, which is a dead end, or loop back to `cell_A`, causing infinite recursion.

The purpose of this example is to demonstrate why directed fuzzing is challenging. Even in this simple program, there are several complexities:

1. **Control Flow Dependencies**: Understanding the caller and callee relationships among functions is essential. This involves analyzing the program's control flow graph (CFG) to identify paths that can reach the target.
2. **Data Flow Dependencies**: Conditions in the CFG may depend on variables whose values are determined by previous computations or inputs not on the direct path.
3. **Path Condition Solving**: Generating inputs that satisfy all the path conditions involved is crucial.

In our example, to reach `cell_D` (our target), a test input must satisfy multiple conditions:

- Must be greater than 50 (condition in `cell_A`).
- Must be divisible by 7 (condition in `cell_C`).

<!-- Traditional fuzzers face two major inefficiencies here:

1. **Random Exploration**: A fuzzer might waste significant time generating inputs that don't satisfy these conditions at all.
2. **Lack of Fine-Grained Control**: Even when a fuzzer successfully generates an input that reaches `cell_C` (a step closer to the target), its next mutation is still random. This means it might generate a new input that falls back to `cell_B` (a dead end), losing the progress it just made.

Think of it like trying to solve a maze while blindfolded—even if you're told you're getting closer to the exit, your next step is still a guess that might take you backwards. -->

## Understanding the Testbed

To systematically evaluate how LLMs perform on directed fuzzing tasks, we need a controlled testing environment. This is where the Fuzzle benchmark comes in. Fuzzle synthesizes the bug method into a maze-generation process, where triggering a bug is analogous to finding the maze's exit. Successfully solving the maze demonstrates a model's ability to generate correct inputs that reach the buggy function.

In this study, we used a 10x10 maze program generated by Fuzzle because. Think of it as a more complex version of our previous example, but instead of 4 cells (`cell_A` to `cell_D`), it has 100 functions (`func_0` through `func_99`). Here's what makes it interesting. Based on preliminary results, only the OpenAI o1 model performed relatively well. Testing larger and more complex maze sizes seemed unnecessary.

1. **Program Structure**:
   - Each function represents a cell in the maze.
   - Function calls represent connection paths between cells.
   - The program starts at `func_0` (entrance).
   - `func_bug` represents our target (exit).

2. **Navigation Rules**:
   - Each function reads one byte from the input.
   - Based on this byte's value, it decides which function to call next.
   - Wrong decisions lead to dead ends or infinite loops.

The program structure can be visualized as a maze:

{% include figure.html path="assets/img/2025-04-28-can-llms-help-directed-fuzzing/maze_viz.png" class="img-fluid" %}

- **White cells**: Functions in the program.
- **Black cells**: No connection exists between functions.
- **Yellow cells**: Critical path functions that must be traversed to reach the target.
- **Entry point (top-left)**: `func_0`, where program execution begins.
- **Target (bottom-right)**: `func_bug`, which we aim to reach.

This visualization helps us understand why directed fuzzing is challenging:

1. The maze is large (100 cells vs. our 4-cell example).
2. Many paths lead to dead ends.
3. The correct path requires satisfying multiple conditions.
4. Some paths may loop back to previously visited functions.

### Human Expert Chain-of-Thought

To trigger the bug, a human expert would approach this problem through the following chain-of-thought:

1. **Analyze Function Relationships**: Determine caller and callee relationships among all functions.
2. **Identify the Critical Path**: Figure out the exact path marked by the yellow cells:

   ```
   func_0 -> func_10 -> func_11 -> func_12 -> func_13 -> func_14 -> func_24 -> func_34 -> func_44 -> func_43 -> func_42 -> func_32 -> func_31 -> func_41 -> func_51 -> func_52 -> func_62 -> func_72 -> func_82 -> func_83 -> func_93 -> func_94 -> func_84 -> func_74 -> func_75 -> func_76 -> func_86 -> func_87 -> func_88 -> func_78 -> func_79 -> func_89 -> func_99 -> func_bug
   ```

3. **Collect Path Conditions**: Along this path, collect all the conditions that must be satisfied based on the input bytes read at each function.
4. **Solve the Path Conditions**: Generate the final input that satisfies all these conditions to reach `func_bug`.

## Experiment Results

### Selected LLM Models

We evaluated popular LLM models accessible through web interfaces, including:

- **OpenAI**: o1, o1-mini, GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Google**: Gemini 1.5 Pro

### Prompts

**Zero-Shot Prompt**

We tested whether the LLMs could, without any hints or prior context, directly provide an input that reaches `func_bug`. We provided the test program by copying and pasting the source code into the chatbox, followed by the question:

```
Examine the uploaded code. Provide a call trace starting from the main function to 'func_bug'.
```

To clarify potential confusion due to local variables with the same name, we added:

```
Please note the variable 'c' is a function local variable that only changes with each function call and is determined by the input at the current index.
```

**Feedback-Driven Approach**

If the LLM's initial response was incorrect, we provided feedback indicating the error and asked them to fix it.

**Chain-of-Thought (CoT) from o1**

We provided the chain-of-thought generated by o1 to other LLMs to see if it would help them produce the correct answer.

**[Human Expert Chain-of-Thought (CoT\*)](#human-expert-chain-of-thought)**

We supplied the detailed reasoning steps from a human expert to the LLMs.

### Testing Methodology

We designed the testing methodology as follows:

1. **Zero-Shot Evaluation**: Test if the LLM can directly provide the correct input without any hints.
   - If **yes**, record success and stop testing.
   - If **no**, proceed to steps 2 and 3 simultaneously.
2. **Using o1's CoT**: Provide OpenAI o1's chain-of-thought to other LLMs and see if they can produce the correct input.
   - If **yes**, record success and stop testing.
   - If **no**, proceed to step 4.
3. **Feedback-Driven Correction**: Inform the LLM whether their output is correct or not, hoping it can self-correct and provide the answer.
   - If **yes**, record success and stop testing.
   - If **no**, proceed to step 4.
4. **[Using Human Expert CoT**](#human-expert-chain-of-thought): Provide the human expert's chain-of-thought to the LLMs.
   - If the LLM makes any mistakes, record failure.
   - If the final answer is correct, record success.

### Results Summary

We have redesigned the table to break down the problem into three evaluation criteria:

1. **Correct Caller-Callee Relationships**
2. **Correct Call Sequence**
3. **Correct Input**

Below is the table summarizing the performance of each model across the different testing methods and evaluation criteria. A checkmark (✔) indicates success, a cross (✖) indicates failure, and N/A indicates that the test was not applicable.

| **Test Method**          | **Evaluation Criteria**        | **o1** | **o1-mini** | **GPT-4** | **GPT-4o** | **GPT-4o-mini** | **Claude 3.5 Sonnet** | **Claude 3 Haiku** | **Gemini 1.5 Pro** |
|--------------------------|-------------------------------|--------|-------------|-----------|------------|-----------------|-----------------------|--------------------|--------------------|
| **Zero-Shot Prompt**     |  Correct Caller-Callee      | ✔      | ✔           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Call Sequence      | ✔      | ✖           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Input              | ✖      | ✖           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
| **Feedback-Driven**      |  Correct Caller-Callee      | N/A    | ✖           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Call Sequence      | N/A    | ✖           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Input              | ✔      | ✖           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
| **Using o1's CoT**       |  Correct Caller-Callee      | N/A    | ✔           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Call Sequence      | N/A    | ✔           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Input              | N/A    | ✖           | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |
| **Using Human Expert CoT** |  Correct Caller-Callee    | N/A    | N/A         | ✖         | ✔          | ✖               | ✔                     | ✔                  | ✔                  |
|                          |  Correct Call Sequence      | N/A    | N/A         | ✖         | ✔          | ✖               | ✖                     | ✖                  | ✖                  |
|                          |  Correct Input              | N/A    | N/A         | ✖         | ✖          | ✖               | ✖                     | ✖                  | ✖                  |

### Observations

#### [o1](#gpto1-chat-dialog)

From the chat dialog, o1 can break down the problem into reasonable subtasks. It found the correct call trace to the target `func_bug`. Examining its chain-of-thought, we see that o1 effectively navigated the maze, avoided dead ends and loops, and collected the necessary path conditions. However, it initially provided an incorrect input array due to miscounting indices.

Upon informing o1:

```
The function call trace is correct; however, the input array is wrong. Figure out why and fix it.
```

After 2 minutes and 3 seconds of processing, o1 realized the mistake and successfully corrected the error, ultimately providing the correct input.

#### [o1-mini](#gpto1-mini-chat-dialog)

While o1-mini attempted to analyze the problem, it incorrectly concluded that there is no path from `func_0` to `func_bug`. When provided with o1's chain-of-thought, it was able to find the correct path but faced similar issues with variable confusion. Even after feedback, it couldn't resolve the problem.

#### [GPT4o](#gpt4o-chat-dialog), [GPT4o-mini](#gpt4o-mini-chat-dialog) and [GPT4](#gpt4-chat-dialog)

GPT-4 and its variants struggled with hallucinations and could not correctly determine the caller and callee relationships. Even when given the human expert's chain-of-thought, only GPT-4o was able to understand and produce the correct call relationships but struggled with solving the path conditions.

#### Claude 3.5 and Gemini 1.5 Pro

Although these models could identify some function relationships, they suffered from hallucinations, such as assuming `func_15` calls `func_21`, which was incorrect. They could not find the correct call trace.

### Comparison with Traditional Directed Fuzzers

We also evaluated state-of-the-art directed fuzzers on this maze program:

|                              | **AFLGo</d-cite><d-cite key="bohme2017directed"></d-cite>** | **AFL++<d-cite key="fioraldi2020afl"></d-cite>** | **DAFL<d-cite key="kim2023dafl"></d-cite>** | **SelectFuzz<d-cite key="luo2023selectfuzz"></d-cite>** | **Beacon<d-cite key="huang2022beacon"></d-cite>** |
|------------------------------|-----------|-----------|----------|----------------|------------|
| **Time to Exposure (min)**   |   4.97    |   0.15    |   0.17   |      3.25      |    1.38    |

**Key Findings**:

1. Traditional directed fuzzers can solve this maze program more efficiently than LLMs.
2. High throughput allows fuzzers to solve the maze without reasoning, relying on rapid random input generation.

## Ways That LLM Might Help

Our experiments revealed several insights about using LLMs for directed fuzzing:

**Pros**:

1. **Semantic Understanding**: LLMs demonstrated a superior ability to understand program semantics compared to traditional static analysis, which may overlook implicit control and data flow.
2. **Human-Like Reasoning**: LLMs approach problems more like humans, making decisions at a higher level rather than relying solely on predefined algorithms.

**Cons**:

1. **Performance**: LLMs are much slower compared to traditional static and dynamic analysis tools.
2. **Context Window Limitations**: Large programs exceed current LLM context windows, requiring complex chunking strategies.
<!-- 3. **Consistency**: LLM outputs can be non-deterministic, potentially leading to inconsistent guidance. -->
4. **Hallucinations**: LLMs may generate incorrect or fabricated information.

Recent academic work</d-cite><d-cite key="chang2024leveling"></d-cite> has shown promising results by combining LLMs with fuzzing techniques, particularly in automatic generation of fuzz targets and LLM-assisted input generation. Leveraging the reasoning and decision-making capabilities of LLMs, future directions could include:

1. **Directed Fuzz Introspection**: Using LLMs to debug and identify bottlenecks in the fuzzing process and provide feedback to the fuzzer.
2. **Assisting Static Analysis**: Filtering out false positives and negatives in traditional static analysis.
3. **Path Condition Solving**: Serving as solvers for complex path conditions.
4. **Strategy Generation**: Generating high-level strategies for exploration, complemented by low-level program analysis tools.

## Appendix

### Maze Program Source Code

<details>
<summary>View Code</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/Wilsons_10x10_0_1_25percent_default_gen.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### o1 Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/gpto1-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### o1-mini Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/gpto1-mini-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### GPT-4 Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/gpt4-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### GPT-4o Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/gpt4o-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### GPT-4o-mini Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/gpt4o-mini-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### Claude 3.5 Sonnet Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/claude-3.5-sonnet-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### Claude 3 Haiku Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/claude-3-haiku-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>

### Gemini 1.5 Pro Chat Dialog

<details>
<summary>View Full Dialog</summary>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-can-llms-help-directed-fuzzing/gemini-1.5-pro-dialog.html' | relative_url }}" 
          frameborder='0' 
          scrolling='yes' 
          height="500px" 
          width="100%">
  </iframe>
</div>
</details>