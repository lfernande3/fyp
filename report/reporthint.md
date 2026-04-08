# Report Writing Hints

This file is a drafting guide, not the full report plan. Use it to keep the writing tight, academic, and well ordered while following `reportplan.md`.

## 1. Core Writing Principle

The report should feel like one continuous argument:

1. why the problem matters
2. what was built
3. why the simulator can be trusted
4. what the main results show
5. what those results mean
6. how the extensions broaden the work

If a section does not help that flow, shorten it or move it.

## 2. How to Present the Objectives

Avoid dropping all ten objectives into the prose as a long undifferentiated list. Instead:

- present the grouped structure in Chapter 1
- treat O1-O5 as the main contribution
- present O6-O10 as extension studies after the core results

Recommended wording:

- The project first establishes a validated simulation framework and core design analysis.
- It then extends that framework to several targeted scenarios that test the robustness and scope of the main findings.

## 3. Chapter-by-Chapter Guidance

## Chapter 1 - Introduction and Context

### What this chapter must do

- motivate the M2M delay-lifetime problem
- explain why sleep-aware access matters
- state the research question clearly
- present the grouped objectives
- define scope and report structure

### What to avoid

- spending too long on implementation detail
- introducing every extension too early
- listing figures or tables here

### Good chapter ending

End by telling the reader that the next chapter explains the model, simulator, and experiments that make the analysis possible.

## Chapter 2 - Methodology

### What this chapter must do

- define the system model and key parameters
- introduce the analytical baseline
- explain the simulator architecture
- define metrics and validation logic
- describe the experiment program

### What to emphasize

- how `q`, `t_s`, `t_w`, `n`, and `lambda` enter the model
- why validation comes before optimization
- how reproducibility is maintained

### Good chapter ending

End by telling the reader that the next chapter tests the simulator against theory and then uses it to derive design insights.

## Chapter 3 - Core Results and Design Guidance

### Best internal order

1. validation
2. parameter impact
3. optimization
4. 3GPP-inspired guidance
5. independence analysis

### Why this order works

- validation establishes trust
- parameter sweeps show the raw behavior
- optimization turns behavior into design choices
- 3GPP mapping makes the results practical
- independence analysis delivers the strongest conceptual takeaway

### What to avoid

- mixing extension results into the middle of the baseline story
- overclaiming before the validation evidence is shown

### Good chapter ending

End by saying that the core findings are established and that the next chapter shows how the same simulator was extended to additional scenarios.

## Chapter 4 - Extension Studies

### What this chapter must do

- keep each extension concise and focused
- show what changed from the baseline
- report the main finding in one paragraph per extension

### Suggested structure per extension

- motivation
- method change
- key result
- implication

### What to avoid

- letting extensions overshadow O1-O5
- repeating all baseline methodology in each subsection

### Good chapter ending

End by setting up the discussion chapter as the place where all core and extension insights are interpreted together.

## Chapter 5 - Discussion

### What this chapter must do

- interpret the results rather than restate them
- explain what the coupling between `q` and `t_s` means in practice
- discuss realism, scope limits, and what the 3GPP mapping does and does not claim

### Strong discussion themes

- simulation as a complement to analysis
- trade-off as a design constraint, not a bug
- co-optimization versus sequential tuning
- what the extensions suggest about generality

## Chapter 6 - Conclusion

### What this chapter must do

- summarize the contribution compactly
- revisit the grouped objectives
- separate confirmed findings from future work

### Good final message

The strongest close is not "many things were implemented." The strongest close is that the project produced a validated design-oriented simulator and used it to show that sleep and access decisions must be chosen jointly.

## 4. Transition Hints

Use transitions that move from one chapter purpose to the next:

- Introduction to Methodology:
  The problem definition motivates the need for a model and simulator that can test these trade-offs under controlled assumptions.

- Methodology to Results:
  With the model, implementation, and validation procedure defined, the report can now examine whether the simulator reproduces the baseline theory and what design insights follow.

- Core Results to Extensions:
  After establishing the baseline findings, the same framework can be extended to explore how the conclusions change under richer assumptions.

- Extensions to Discussion:
  Taken together, the core and extension studies clarify both the strength of the main findings and the boundaries within which they should be interpreted.

## 5. Style Hints

- prefer short analytical paragraphs over bullet-heavy prose in the final report
- use objectives to orient the reader, not as the main writing structure
- keep claims proportional to the evidence actually shown in figures and tables
- distinguish clearly between validated findings, design interpretation, and future work
- treat the extensions as support material unless one becomes central to the final thesis

## 6. Final Sanity Check

Before drafting each chapter, ask:

- What is this chapter proving?
- What does the reader need to know first?
- What result or idea should they carry into the next chapter?
