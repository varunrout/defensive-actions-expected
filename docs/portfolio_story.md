# Portfolio story

## Case-study framing

Defensive Actions Expected is a football analytics and data science case study about measuring defensive actions with context. The project starts from a practical football problem: defensive actions are visible in event data, but raw counts are not enough to explain defensive value or danger suppression.

The repository demonstrates how to build a reproducible modelling and analysis workflow before making strong football claims.

## Audience

This story is written for football analytics recruiters, data science reviewers, technical hiring teams and collaborators who want to understand both the football logic and the engineering discipline behind the project.

## What the project demonstrates

### Modelling

The project includes classification for future shot probability, regression for future xG, two-part future-xG signals, OOF predictions and sensitivity-model planning. It shows how predictive modelling can be connected to football questions without pretending that every prediction is a causal value estimate.

### Football logic

The analysis is built around football-defined populations such as centre-back own-box actions rather than generic rows alone. It recognises that location, role, phase, possession context and future sequence outcomes matter when interpreting defending.

### Validation discipline

The project emphasises OOF integrity, coverage checks, duplicate checks, fold-level predictions, metadata completeness and sensitivity warnings. Limitations are surfaced rather than hidden.

### Script-based reproducibility

The coach-analysis layer moved from exploratory notebooks to deterministic scripts. This makes analysis easier to rerun, review, test and extend.

### Coach-facing analysis

The CB box-defence report is designed around interpretable outputs: markdown reports, summary tables, pitch maps, execution summaries and video-review candidate CSVs. The goal is to support football review, not just model scoring.

### Honest limitations

The project is explicit about current gaps: unknown competition metadata in local artifacts, incomplete sensitivity OOF availability, phase/spatial diagnostic mismatch and limited tactical-population coverage. This honesty is part of the portfolio value because it shows judgement and avoids overclaiming.

## Reviewer takeaway

The strongest current takeaway is not “this model has solved defending.” It is that the project builds a credible foundation for threat-aware defensive analysis: deterministic data flow, contextual targets, OOF model outputs, coach-analysis scripts, validation checks and clear next steps.
