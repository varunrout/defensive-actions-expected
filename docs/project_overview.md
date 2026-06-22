# Project overview

## Purpose

Defensive Actions Expected (DAx) is a football analytics project that estimates short-horizon attacking danger after recorded defensive actions. The project focuses on whether the context around a defensive action suggests that the defending team has suppressed danger, delayed danger or left danger unresolved.

The repository has moved away from notebook-first coach analysis and toward deterministic scripts. That shift makes the analysis easier to rerun, test, review and package for a portfolio or production-style workflow.

## Football problem

Raw defensive actions are easy to count but difficult to interpret. A centre-back clearance, block, pressure or duel can occur in very different situations:

- a low-danger touch with many defenders behind the ball;
- an emergency intervention inside the penalty area;
- a pressure that prevents a high-quality shot;
- a tackle that immediately leads to a second-wave chance.

The football question is therefore not only “how many actions did a player make?” but “what danger existed around those actions, and what happened next?”

## Why defensive actions need contextual modelling

Defending is strongly shaped by role, location, game state, possession phase, teammate cover, opposition pressure and ball progression. Counting actions without context can reward players who are repeatedly exposed while under-crediting players whose positioning prevents actions from becoming necessary.

DAx uses contextual features and observed future outcomes to move from activity counts toward threat-aware interpretation. The current project does not claim causal defensive value or invisible defending. It builds the modelling and validation foundation needed before those claims can be made responsibly.

## What “Defensive Actions Expected” means

In this project, “Defensive Actions Expected” refers to expected short-horizon attacking threat after defensive actions. The current modelling layer estimates future shot probability and future xG following a defensive action, then supports analysis of where observed outcomes differ from model expectations.

A useful interpretation is:

- **expected threat after the action:** what the model thinks is likely given the action context;
- **observed threat after the action:** whether a shot or xG actually occurred in the short horizon;
- **diagnostic gap:** where observed outcomes are above or below expectation, subject to validation and sample-size caution.

## Counts versus expected threat and suppression framing

Raw counts answer volume questions such as “how many centre-back own-box defensive actions were recorded?” Expected-threat framing asks whether those actions occurred in more or less dangerous contexts and what short-horizon outcomes followed.

This distinction matters because high action volume is not automatically good or bad. A defender may accumulate actions because the team defends deep, because opponents target that channel, or because the player is excellent at resolving danger. DAx is designed to support this more careful interpretation by combining football-defined populations with modelled future threat and explicit validation checks.
