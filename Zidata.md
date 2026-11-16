# Technical document — possession from automatic tracking

**Submission:** `evaluation/results/possession_state_machine.csv`
**Regeneration:** `python work/model.py` (needs the match data bundle in `data/`)

Public-window scores (17 possessions, 7,500 graded frames):

| | spellF1 (ranking) | macroF1 (tie-break) |
|---|---|---|
| this submission | **0.607** | **0.882** |
| published baselines | 0.038 – 0.424 | 0.597 – 0.621 |

---

## The approach

The system is **two coupled state machines** over the graded frames — one for
*alive/dead*, one for *which team* — driven entirely by hand-designed evidence
with physical units. There is no learned model, no training data, and no
parameter was fitted to a score: every constant is a speed, a distance or a
duration with a documented justification (they are collected at the top of
`work/model.py`).

### 1. The alive/dead state machine

A frame is *dead* when the ball is not being played. The machine enters dead on
three kinds of evidence, each a direct reading of the tracking:

- **Ball out of play** — the ball beyond the pitch lines by a 1 m margin for
  five frames (the margin absorbs tracker noise near the lines).
- **The play has stopped** — for ~1.3 s continuously: the players are not
  moving (rolling mean team speed < 1.5 m/s) *and* the ball is not being
  played (invisible, slower than 1.5 m/s, or more than 6 m from every player).
  This catches the whistle, which no per-frame signal marks directly.
- **Tag-anchored windows** — the Smart Tagging's `GOALS` and kick-off intervals
  are used to force dead through goal celebrations (GOALS tag → the following
  kick-off tag) and through the teams' walk-out before each half's kick-off.

Exiting dead is the harder half. The state machine waits for a **restart kick**:
a ball-speed rise from rest/invisibility through 3.5 m/s, *confirmed* by one of

- **a set-piece tag**: a firm kick (≥ 5 m/s within ten frames) struck from a
  ball at rest inside a `SET PIECES` interval — with the strictness the tag
  type earns. Kick-offs, corners and goal kicks are position-anchored restarts
  the tagger labels reliably and whose kicks the tracker often misses (the ball
  is lost at the flag or in flight), so any struck ball after a rest inside
  those tags confirms. Free-kick and throw-in tags are wide and their setups
  include kick-backs, so they get the full firmness/at-rest/on-pitch checks.
- **play actually resuming**: the ball travels ≥ 15 m within 3 s of the kick
  when it starts near a touchline (a throw-in), or ≥ 20 m *and is still moving*
  at the end of the look-ahead window in open field. The end-of-window check is
  what separates a restart pass from a stoppage kick-back, which is rolled to a
  standstill.
- **an activity escape**: if the ball has been invisible for a while but the
  players are clearly playing (mean speed > 2.5 m/s for 4 s), the machine
  returns to alive — the tracker lost the ball, the game did not stop.

### 2. The team state machine

Team evidence is a **distance-weighted vote** over every visible player:
`vote = Σ exp(−(d/σ)²)`, σ = 2.5 m, one sum per team, taken as a log-ratio. This
is robust to individual slot mislabels in a way the nearest-player rule is not
(46 track slots hold no identity and occasionally carry the wrong team).

The vote is smoothed (±0.4 s), then decoded inside alive spans with three
rules that implement the challenge's possession semantics:

- **A flip commits only on decisive, sustained evidence**: the log-ratio must
  cross ±2.5 (a ~12:1 weighted vote) and stay beyond it for 70% of the next
  2 s. Transient excursions (a defender challenging, tracker noise) are
  cancelled instead of becoming spurious spells. The flip is placed at the
  crossing frame, so confirmed flips are not delayed by the confirmation.
- **No controlled change while the ball is in flight**: a flip cannot commit
  while the ball is moving faster than 7 m/s. Loose and flying balls keep the
  previous owner, exactly as the ground truth does.
- **The kicking team owns the restart**: in the first second after a restart,
  if the ball is at a player's feet (nearest player < 2 m) and the vote is
  decisive, the team is set to the vote's side without the hold — the kicker
  is visible at the kick.

### 3. The broadcast recording as a second opinion

The broadcast ATD is an independent tracking pipeline on a different camera,
so its vote is a second observation of the same event. Its timeline was aligned
to the tactical one by cross-correlating the two ball trajectories over the 8
dihedral pitch transforms: the identity transform at offset +4,010 frames fits
with a median ball-position error of ~1.4 m, stable across the whole match
(no drift; `work/diag12.py`). Broadcast frames whose aligned ball disagrees
with the tactical ball by more than 6 m (replays, cuts) contribute nothing.

The fusion is deliberately **asymmetric**: where the broadcast vote agrees in
sign with the tactical vote *and the tactical vote is undecided* (|logit| < 2.5),
the broadcast vote is added — it can tip a balance, never dilute a decision.
Averaging the two unconditionally hurt (it delayed real flips); so did fusing
the smoothed signals (it let a brief agreement flip a state back). The final
form excites the raw per-frame votes before smoothing.

## Which assets were used, and how

| asset | use |
|---|---|
| **tactical ATD** (txt + xml) | everything: ball, players, team slots, speeds |
| **Smart Tagging** (xml) | `SET PIECES` intervals as restart priors; `GOALS`/kick-off intervals as forced-dead windows. Phases with team tags were evaluated as team evidence and *dropped* — they disagreed with the vote too often to be worth the complexity |
| **broadcast ATD** (txt + xml) | second vote after timeline alignment |
| **homography** (both) | not needed: possession uses pitch coordinates only; the broadcast homography is never consulted |
| **video** (both) | not used |
| `kloppy` | used only in the starter notebooks; the pipeline reads the files directly |

**Training data: none.** No labelled dataset of any kind was used, from this
game or any other. No hand labels. The public 5-minute ground truth was used
only to check that the pipeline was correct and to understand what the
definitions mean in practice, as the challenge prescribes — it is never read
by the pipeline (`work/model.py` has no dependency on the ground-truth files).

## The possession rules implemented

- A possession is a maximal run of one team with the ball alive; dead-ball
  stretches split spells even of the same team (the metric's own definition).
- The ball is dead exactly while the state machine says so: out of play,
  play stopped, goal celebrations, the pre-kick-off walk-outs.
- A change of possession requires the challenger's vote to be decisive *and*
  sustained, and is impossible while the ball is in flight — loose and
  contested balls keep the previous owner, per the challenge definition.
- Restarts give possession to the kicking team immediately (restart-set rule
  plus the tag prior), because the ground truth assigns the kicker the ball
  from the strike.
- `X` is written outside the graded blocks only; inside them the model always
  answers.

## Limitations

- **Short spells in rapid exchanges.** The ground truth contains possessions of
  0.8–3.6 s that the uncorrected vote barely sees (the corrected tracking that
  produced the labels is better than either provided tracker). These are
  missed; the decoder is deliberately conservative because a spurious flip
  costs more than a missed one under spellF1.
- **Sustained tracker errors.** Where the tactical tracker carries a wrong ball
  position or wrong slot teams for many seconds (both trackers agree on one
  ~13 s stretch in the public window that the ground truth labels otherwise),
  the model follows the tracking.
- **Whistle timing.** A foul is marked dead when the players visibly stop;
  that is 1–3 s after the whistle.
- **Soft untagged restarts** (a short throw-in with no tag and little
  displacement) are detected late, at the next firm pass.
- With more time: a vision model on the video for the ball (the single most
  valuable missing signal), a learned slot-team correction, and a proper
  probabilistic duration model instead of the hold rule.

## Reproduction

```bash
pip install -r requirements.txt        # numpy, pandas
python work/model.py                   # writes evaluation/results/possession_state_machine.csv
python evaluation/validate_submission.py evaluation/results/possession_state_machine.csv
python evaluation/run_evaluation.py    # scores every csv in evaluation/results/
```

The run needs the gated data bundle in `data/` (see `DATA.md`). It takes a few
minutes; feature caches under `work/cache/` (gitignored — derived tracking
data) make repeat runs faster. The output is deterministic.
