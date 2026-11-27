# Using these files in Overleaf (IEEE Access template)

1. Upload `sec1_introduction.tex`, `sec2a_related_fraud.tex`, `sec3_dataset.tex`,
   `sec4_methodology.tex`, `summary_table_rows_member1.tex`, the `tables/` folder and the
   `figures/` folder to the Overleaf project.
2. In the template's main file, add `\usepackage{array}` (used by the feature-group table) and
   put `\input{sec1_introduction}` etc. where the sections go.
3. Merge `references_member1.bib` into the group's .bib file; keep `\bibliographystyle{IEEEtran}`.
4. Section II: Member 2 adds `\subsection{Concept Drift}\label{sec:related-drift}`, Member 3 adds
   `\subsection{Algorithmic Fairness}\label{sec:related-fair}`; add your rows to the summary table.
5. Labels used for other members' sections: sec:results, sec:fairness, sec:ablation,
   sec:discussion, sec:conclusion. Use these names or change them in sec1 and sec4.
6. Red text from `\tbd{...}` marks numbers that are filled in from the EDA output. Define
   `\providecommand{\tbd}[1]{#1}` in the preamble (or remove the wrapper once filled).
