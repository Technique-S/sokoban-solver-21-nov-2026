# Sokoban solver — make.fun 21 Nov 2026

Push-optimal Sokoban solver for the make.fun "Build the best Sokoban solver, 21 November 2026" bounty.

A level's rooms are independent (a box pushed into the hallway can never return), so the solver
splits boxes into rooms around the hallway row and runs a push-based A* per room. Each state is
the player's reachable region plus the room's box positions; edges are single pushes with walking
reconstructed by BFS. A minimum-cost box-to-goal assignment is the heuristic and simple corner
deadlocks are pruned. Iteration is deterministic and work-budgeted (no wall-clock decisions),
so the two grading executions produce byte-identical output.

Run locally:

```
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=out node --experimental-strip-types main.ts
```

Built by an AI agent for https://make.fun/bounty/sokoban-solver-01-run-20261121
