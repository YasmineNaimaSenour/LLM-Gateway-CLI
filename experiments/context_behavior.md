# Context Window Behavior — Experiment Log

## Goal
Observe how a given provider/model behaves as the input context grows, and identify:
- the point where quality degrades noticeably
- the point where the request fails outright (`CONTEXT_OVERFLOW`)
- any silent truncation behavior

## Setup
| Field | Value |
|---|---|
| Provider | |
| Model | |
| Context window (per model card) | |
| Prompt template | |

## Trials
| # | Input tokens (approx.) | Status | Observed behavior | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## Observations
-

## Failure point
- Tokens at which `CONTEXT_OVERFLOW` was first triggered:
- Error message returned:

## Conclusions
-
