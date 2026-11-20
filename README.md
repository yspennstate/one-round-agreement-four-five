# Optimal one-round agreement through stochastic channels for four and five processes

Code, data and check outputs for the paper of the same title (`paper/main.tex`, `paper/main.pdf`).

## The problem

Processes with binary inputs exchange one round of messages, each delivered independently with
probability p, and then decide values in [0,1]. When all inputs agree, every process must decide the common
input. The cost of an algorithm is its expected discrepancy (the largest difference between two decisions)
on the worst input assignment. Fraigniaud, Patt-Shamir and Rajsbaum (DISC 2025, doi:10.4230/LIPIcs.DISC.2025.32)
found the optimal one-round algorithms for two and three processes and asked for optimal algorithms for any
number of processes n > 2. The paper determines the optimal one-round cost for four and five processes and
every p. At p = 1/2 the optima are 941/1936 (n = 4) and 394955/851328 (n = 5).

## Layout

- `code/` - the scripts and the matrices `hull_M_n3.npz`, `hull_M_n4.npz`, `hull_M_n5.npz`. Run everything from inside
  `code/`; the scripts read the matrices from the working directory.
- `outputs/` - the output of every check reported in the paper.
- `code/certs/` - the 61 exact subgradient certificates that give a second proof of the n = 4 lower bound
  above alpha_4, one for each cell met by the optimal path; `outputs/cert_verify_all.txt` is the verifier's log and
  `outputs/cert_cover_*.txt` record how the intervals were covered.
- `paper/` - LaTeX source and PDF.

Requirements: Python 3.13 with sympy 1.14, numpy 2.5 and scipy 1.18 (tested with these versions).

## What each script does

| script | what it checks | output |
|---|---|---|
| `hull.py n` | builds the integer matrices M_k: the expected discrepancy of every decision vector in {0,1/2,1}^m, as a polynomial in p | `hull_M_n<n>.npz`, `outputs/hull_out.txt`, `outputs/hull3.txt` |
| `regcheck.py n [NEG] [regime]` | the exact lower bound: lambda lies in [0,1], and lambda E_1 + (1-lambda) E_2 >= OPT at every vertex, by Sturm sequences and algebraic root isolation. `NEG` in {opt, mu, ext} runs a deliberately wrong version that must fail; `-` means none | `outputs/rc3.txt`, `rc4.txt`, `rc5_*.txt`; negative controls `rc3_neg.txt`, `rc4_neg_*.txt`, `rc5_neg.txt` |
| `half5.py [opt]` | the n = 5 lower bound at p = 1/2 in exact rationals over all 3^10 vectors; with `opt` the value is raised by 1/1000 and the check must fail | `outputs/half5.txt`, `half5_neg.txt` |
| `n5regimes.py` | the n = 5 regimes: closed forms of OPT_5, lambda and the thresholds | used by `regcheck.py`, `ub5.py` |
| `ub5.py` | the n = 5 upper bounds: an explicit algorithm attaining OPT_5 on every interval | `outputs/ub5_out.txt` |
| `verify_exact.py`, `mix4.py` | the n = 4 closed forms and the two-vertex mixtures; OPT_4(1/2) = 941/1936 | `outputs/verify_exact.txt`, `mix4.txt` |
| `certdump.py`, `cert_fixed.py` | build a certificate at a point p0 for the moving and the fixed optimal algorithm, and the largest interval on which it is valid | `code/certs/*.json` |
| `cover.py FAMILY start end` | walks from start to end, calling `certdump.py` at the point just past the end of each valid interval | `outputs/cert_cover_*.txt` |
| `verify_cert.py certs/<file>.json ...` | checks a certificate exactly in Q(p), rebuilding E_1 and E_2 from all 2^12 delivery patterns; its evaluator is also used by `indep.py` | `outputs/cert_verify_all.txt` |
| `indep.py`, `indep5.py`, `indep5r.py` | independent recomputation of rows of M_1 and M_2 by enumerating every delivery pattern, with no grouping | `outputs/indep.txt`, `indep5.txt`, `indep5r.txt` |
| `n3check.py` | the same pipeline reproduces the n = 3 optimum of Fraigniaud, Patt-Shamir and Rajsbaum | `outputs/n3check.txt` |
| `phi_poly_check.py` | Phi_1 and Phi_1/2 are rational functions that are not polynomials | `outputs/phi_poly_check.txt` |
| `paperfacts.py` | every number and polynomial printed in the paper, recomputed | `outputs/paperfacts.txt` |

Examples:

    cd code
    python regcheck.py 4          # n = 4 lower bound on all four intervals: ALL OK
    python regcheck.py 4 mu       # negative control, lambda raised by 1/100: SOME FAILED
    python regcheck.py 3          # n = 3, Theorem 13 of Fraigniaud, Patt-Shamir and Rajsbaum: ALL OK
    python ub5.py                 # n = 5 upper bounds: UPPER ALL OK
    python indep5.py 1110111000 1111111000 hhhhhhhhhh hhh1hhhhhh hh11hhhhhh h1110h10h0 h1110h1000 h1110hh0h0
    python indep5r.py 400 20261117 1110111000 1111111000 hhhhhhhhhh hhh1hhhhhh hh11hhhhhh h1110h10h0 h1110h1000 h1110hh0h0

A vertex is written as one character per view (`0`, `h` for 1/2, `1`). The n = 5 lower bound is split into
regimes, run one at a time: `python regcheck.py 5 - R` for R in `low`, `AMP`, `P1`, `P2`, `P3`, `P4`, `sliver`.
The seven intervals are [0, 0.109884], [0.109884, 0.410601], [0.410601, 0.414398], [0.414398, 1/2],
[1/2, 0.565484], [0.565484, 0.673303] and [0.673303, 1]. On one core of a Mac mini, `low` and `AMP` take about a
minute each and `sliver` about 2.6 hours; `P1` to `P4` take 21 to 29 hours each (the seconds are printed in each output
line). `outputs/rc5_*.txt` are the runs on one machine. `P1`, `P2` and `P3` were run again on a second machine from a
fresh copy of the code, with the same sha256 for `regcheck.py` and `hull_M_n5.npz`; those outputs are in
`outputs/second_machine/`. `P4` was run on one machine.

## Citation

If you use this code, please cite the paper.
