---
description: "Use when editing or generating LaTeX course notes, math lecture handouts, or formula-heavy .tex files in this repository. Enforces notation consistency for neural-network formulas, transpose notation, and OCR/AI formula cleanup."
name: "LaTeX Formula Conventions"
applyTo: "**/*.tex"
---

# LaTeX Formula Conventions

- Before adding or rewriting formulas in an existing section, inspect nearby earlier content and reuse the established notation instead of introducing a new symbol system.
- For layer-wise feedforward neural network notation, use `\boldsymbol{W}^{(l)}`, `\boldsymbol{b}^{(l)}`, `\boldsymbol{z}^{(l)}`, and `\boldsymbol{a}^{(l)}` for matrix or vector forms. Keep scalar or elementwise symbols unbolded only when the surrounding context is explicitly scalar.
- Use `\delta` for the backpropagation error term. 
- Do not replace mathematically distinct notation such as the Kronecker delta `\delta_{ij}` when it is not the backpropagation error term.
- Always write transpose as `^\top`, never `^top`.
- When formulas appear to come from OCR or AI transcription, verify them and correct obvious notation or mathematical errors before keeping the generated text.
- Keep edits local to the affected teaching material and update the adjacent explanatory prose when notation changes make the existing sentence inconsistent.