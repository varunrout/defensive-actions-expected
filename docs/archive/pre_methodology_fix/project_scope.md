# Defensive Actions Expected (DAx) — Project Scope, Objective, and Need

## 1. Executive summary

**Defensive Actions Expected (DAx)** is a football analytics project designed to measure defending as the **suppression of attacking futures**, not as a simple count of defensive events.

Traditional defensive metrics focus on visible actions such as tackles, interceptions, clearances, blocks, and pressures. Those events are useful, but they miss a lot of what makes defending effective:

- a defender closing a passing lane before the pass is attempted
- a back line holding the pitch compact enough to remove central access
- a midfielder screening the cutback lane without touching the ball
- a winger delaying a transition so teammates can recover
- a defensive unit shifting together to reduce the quality of the attacker’s choices

DAx is built to answer a different question:

> **How much attacking threat was prevented, delayed, redirected, or removed by the defending team and by each defender?**

This project is intentionally designed for practical football analysis. It should produce outputs that are understandable to analysts, coaches, and data scientists while remaining grounded in match data that can actually be obtained from StatsBomb open data and 360 freeze-frame data where available.

---

## 2. Why this project is needed

### 2.1 The problem with traditional defensive metrics

Most conventional defensive metrics are event-based. They count actions that are recorded explicitly in the data:

- tackles
- interceptions
- blocks
- clearances
- pressures
- duels won
- recoveries

These are valuable, but they do **not** fully represent defending because they only measure what happened, not what was prevented from happening.

A defender can have strong impact without recording a tackle:

- by standing in the right lane
- by forcing a backwards pass
- by delaying a progressive carry
- by preventing a switch of play
- by protecting the box before the cross is even delivered
- by helping the team absorb pressure without allowing a high-value shot

This means event counts can undervalue intelligent, invisible, or structural defending.

### 2.2 What DAx is trying to capture

DAx is intended to measure defensive influence in terms of **option control**:

- what passes were available
- what carries were available
- what central or wide routes were open
- which runners could be accessed
- how dangerous the next few attacking actions could become
- how much the defensive structure reduced those options

The idea is not only to reward ball-winning actions, but also to reward the defenders who make those actions unnecessary.

### 2.3 Why it matters

This project is useful because it can:

- identify defenders whose value is undercounted by standard stats
- distinguish active suppression from passive defending
- compare individual defenders across roles and teams more fairly
- quantify team defensive structure, not just individual events
- separate high-quality pressure from bad pressure
- capture counterpressing, box defence, transition recovery, and lane blocking in a coherent framework

---

## 3. Project objective

### 3.1 Core objective

The objective of DAx is to build a football analytics system that measures:

1. **Player Defensive Actions Expected**
   - how much threat an individual defender suppresses through positioning, pressure, lane blocking, space occupation, runner control, anticipation, and recovery

2. **Team Defensive Actions Expected**
   - how much threat a team defensive structure suppresses through compactness, coverage, rest defence, counterpressing, box protection, wide traps, and collective interaction

### 3.2 Formal working definition

A simple working definition is:

> **DAx = expected attacking threat before defensive influence − expected attacking threat after defensive influence**

This can be estimated at the event level, frame level, or possession-situation level.

### 3.3 What success looks like

The project is successful if it can consistently:

- distinguish dangerous attacking situations from harmless ones
- identify defensive actions that reduce future threat even when no explicit event is recorded
- produce player and team values that make football sense
- align with video review and tactical interpretation
- remain stable across competitions and match contexts
- support both descriptive analysis and future modelling work

---

## 4. Scope of the project

### 4.1 In scope

DAx will focus on international tournament football using StatsBomb open data:

- **Euro 2020**
- **FIFA World Cup 2022**
- **Euro 2024**

The project will use:

- competitions data
- match data
- event data
- lineups
- shot data
- possession chains derived from events
- 360 freeze-frame data where available

The model will cover the following analytical layers:

#### A. Defensive theory and definitions
- threat suppression
- option removal
- defensive presence
- anticipatory spatial control
- defensive option control
- transition contingency control
- counterpress threat suppression
- box threat suppression
- deep-block threat exchange control
- team interaction value

#### B. Data acquisition and preparation
- download and store raw StatsBomb JSON
- transform to clean tabular datasets
- attach match, possession, event, and freeze-frame context
- create processed parquet tables

#### C. Defensive phase segmentation
- high press
- counterpress after loss
- transition defence
- settled mid-block
- settled low block
- box defence
- wide defending
- central progression defence
- rest-defence situations
- second-ball situations

#### D. Attacking threat baseline
- baseline attacking threat score
- threat by zone, action, and phase
- simple xT-style modelling and improved supervised variants

#### E. Option tree and counterfactual logic
- possible future attacking actions
- option feasibility and openness
- branch-level threat values
- branch-level responsibility

#### F. Player and team defensive features
- pressure features
- lane-blocking features
- space occupation features
- runner-control features
- recovery and contingency features
- team shape and interaction features

#### G. Defensive suppression modelling
- rule-based baseline
- supervised suppression model
- counterfactual marginal value
- interaction value
- phase-specific value

#### H. Attribution and outputs
- player DAx
- team DAx
- pair and unit interaction values
- phase-specific outputs
- dashboards, visualisations, and case studies

### 4.2 Out of scope for the first version

To keep the project practical, the first versions will **not** rely on:

- proprietary tracking data
- full optical tracking coordinates for all matches
- player communication data
- coaching staff annotations
- post-match tactical labels from manual video coding at scale

The project may later be extended to include these, but they are not required for a strong first release.

---

## 5. Football logic behind the project

DAx is based on a simple football idea:

> Good defending is often the removal of good attacking choices before the attack can exploit them.

A defender can help by:

- closing a direct route to goal
- screening a receiver between the lines
- protecting the space behind a pressing teammate
- controlling a runner before the pass is played
- forcing the ball wide into lower-value space
- delaying a transition so the team can recover shape
- reducing the quality of the next action even without a duel

This means defending should be measured as a combination of:

- **pressure**
- **positioning**
- **coverage**
- **compactness**
- **timing**
- **coordination**
- **anticipation**

DAx attempts to turn those ideas into measurable quantities.

---

## 6. Data requirements

### 6.1 Required datasets

The project needs the following StatsBomb data tables:

- **competitions**: identify tournaments and seasons
- **matches**: identify match context and participants
- **events**: event-level match actions and timestamps
- **lineups**: player participation and roles
- **shots**: shot outcomes and xG-related targets
- **360 freeze-frames**: player and ball spatial context when available

### 6.2 Required identifiers and links

The modelling data should preserve links between:

- `competition_id`
- `match_id`
- `possession_id`
- `event_id`
- `timestamp`
- team in possession
- defending team
- event type
- ball location
- player locations
- freeze-frame visibility flags
- teammate/opponent labels

### 6.3 Data storage targets

The repository should store data in a layered structure:

- `data/raw/` for original JSON files
- `data/processed/` for enriched parquet tables
- `data/features/` for engineered feature outputs
- `outputs/models/` for trained artifacts
- `outputs/validation/` for checks, summaries, and diagnostics
- `outputs/oof/` for out-of-fold prediction artifacts

### 6.4 Why raw and processed layers matter

A separate raw layer is essential because it keeps the original source data intact.

A processed layer is essential because it makes modelling reproducible and efficient.

This separation supports:

- traceability
- reproducibility
- debugging
- future reprocessing
- model versioning
- clean experimentation

---

## 7. Project phases and delivery logic

### Phase 1 — Project definition and defensive theory
Define the football problem, set the scope, and translate defensive concepts into measurable terms.

### Phase 2 — Data acquisition and preparation
Pull StatsBomb data, build clean tables, and create the modelling schema.

### Phase 3 — Possession and phase segmentation
Label the defensive context of each situation so that different game states can be analysed separately.

### Phase 4 — Attacking threat baseline
Measure the danger of an attacking state before defensive influence is credited.

### Phase 5 — Possible futures and attacking option tree
Estimate what attacks could happen next, not only what actually happened.

### Phase 6 — Player defensive feature engineering
Build defender-level features for pressure, lane blocking, coverage, and recovery.

### Phase 7 — Team defensive feature engineering
Build collective structure features such as compactness, line spacing, and rest defence.

### Phase 8 — Defensive suppression model
Estimate threat removed, delayed, or downgraded by defenders and team structure.

### Phase 9 — Attribution framework
Assign defensive value fairly across individuals, pairs, units, and teams.

### Phase 10 — Phase-specific models
Build specialized models for settled defence, counterpressing, transition defence, and box defence.

### Phase 11 — Validation and football sanity checks
Test the model statistically and verify that the outputs make tactical sense.

### Phase 12 — Outputs, dashboards, and storytelling
Create player and team visuals that explain the value of defensive actions.

### Phase 13 — Final project packaging
Package the whole project as a strong portfolio and research asset.

---

## 8. Minimum viable version versus advanced version

### 8.1 Minimum viable version

The first useful version of DAx should include:

- StatsBomb open-data ingestion
- clean event and freeze-frame tables
- defensive phase labels
- a basic attacking threat baseline
- simple rule-based defensive suppression
- initial player and team summaries

This version is enough to start building insight and validating the concept.

### 8.2 Intermediate version

The next version should add:

- candidate attacking option trees
- player-level defensive feature engineering
- team structure features
- supervised suppression modelling
- phase-aware attribution

### 8.3 Advanced version

The advanced version should include:

- counterfactual simulations
- interaction modelling
- phase-specific models
- richer validation and case studies
- dashboard and portfolio presentation

---

## 9. Deliverables

The project should ultimately produce:

- a clean data pipeline
- reusable feature engineering scripts
- phase segmentation logic
- an attacking threat baseline model
- defensive suppression scores
- player and team rankings
- validation reports
- visual analysis notebooks
- a portfolio-grade README and methodology document

---

## 10. Validation criteria

The model should be validated from two sides.

### 10.1 Data science validation
- train/test split by match or competition
- calibration checks
- cross-validation
- baseline comparison
- feature importance analysis
- error analysis by defensive phase

### 10.2 Football validation
- watch clips of high-value defensive sequences
- confirm that invisible defending is credited
- check that passive deep defending is not overvalued
- ensure bad pressure is penalized correctly
- verify that counterpressing and second-ball control are represented
- compare results with football intuition

---

## 11. Risks and limitations

The main limitations are:

- open data does not contain full tracking for every match
- 360 data is not available for every competition or match
- some defensive intent is not directly observable
- attribution is inherently uncertain in complex team defence
- rule-based models may be too simplistic at first

These are acceptable as long as the project is transparent about assumptions and evolves gradually.

---

## 12. Final vision

The long-term vision of DAx is to create a credible and interpretable football analytics model that measures defending as the **suppression of attacking futures**.

The project should feel:

- analytically serious
- tactically meaningful
- reproducible
- transparent
- useful to football analysts
- strong enough to support portfolio presentation and later model extension

That is the standard the project should be built toward from the start.

