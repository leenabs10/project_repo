# Goods diversity failure diagnosis

The saved predictions show a background-specificity failure. Across the three
paired seeds and 12 development frames, the candidate adds 621,842 false-positive
pixels and removes 174,577: a net increase of 447,265. It recovers 118,891 goods
pixels but introduces 79,904 new misses, giving 38,987 net fewer misses. These
are pooled pixel transitions across correlated runs, not object counts or
independent observations. No new inference, fitting or annotation changes ran.

All 264 saved masks were checked: 192 on training references and 72 on development
references. Source identities were verified against the closed experiment.
Ignored pixels remain excluded. The previous utility failure is unchanged.

## Specific observed failures

| Development frame | Visible diagnostic context | Pooled net extra FP | Mean IoU change |
| --- | --- | ---: | ---: |
| 000188 | Near yellow guard and handling-equipment regions | 80,345 | -13.74 pp |
| 000847 | Empty lifting-platform/transport-equipment region | 60,762 | -0.22 pp |
| 000470 | Close empty pallet stack and other equipment | 57,643 | 0.00 pp |
| 000659 | Access stairs, supports and other equipment | 25,410 | -2.60 pp |
| 000282 | Separate wrapped-load miss plus background errors | 24,717 | -10.49 pp |
| 000753 | Tyre coverage improves while background errors increase | 56,852 | +3.13 pp |

Frame 000470 has no foreground reference, so zero IoU in both arms conceals the
different false-positive totals. Ranking only by IoU regression would miss this
important failure. All three seeds add background errors, with net FP increases
of 127,384, 285,768 and 34,113 for seeds 17, 29 and 43 respectively.

About 83.39% of newly false-positive pixels are farther than ten native pixels
from labelled goods or ignored regions, using a Chebyshev band. The share remains
90.43% at five pixels and 72.49% at twenty. This supports larger background-region
errors rather than a predominantly thin mask-boundary problem. It does not
establish the causal training mechanism.

Wrapped-load frame 000282 contains 40,843 gross new missed pixels, offset by
11,354 recoveries. An AI-selected rectangular region around the near wrapped
load contains 37,659 new misses. Rectangles and object descriptions are visual
diagnostic aids, may cover multiple surfaces, and are not new segmentation
annotations. Counting still uses only the unchanged reviewed class IDs.

## Training-side evidence and counterevidence

The five synthetic negatives are camera views of one identical equipment
arrangement. Visible equipment is dominated by wooden pallet components and the
existing red handling-crate source, alongside other background fixtures. The
views do not reproduce the close real guards, lifting-platform and access-stair
appearances implicated in the reviewed failures. Limited negative appearance/view
coverage is a plausible transfer gap, not a demonstrated cause.

The real-only control instead replays training negatives 000482 and 000880
15 additional times each per seed. The synthetic arm replaces those auxiliary
slots with five synthetic empty views, six times each. Primary real exposure
remains 480 in both arms. Both models have zero false positives on the scorable
pixels of those two exact real training negatives. Simple forgetting of those
training examples is therefore not supported.

Tyre frame 000753 improves overall, and dark-goods misses improve despite lower
synthetic dark-pixel coverage. Discarding every synthetic positive or attributing
the failure solely to the dark histogram would overlook this counterevidence.
No demonstrated annotation error explains the observed outcome.

## Proposed next diagnostic

Keep the exact 30 synthetic loaded events, but restore the frozen real training
negative events in the 30 empty slots. Compared with the failed candidate, this
changes negative source/content while retaining synthetic positives. Compared
with the real-only control, it changes positive source/content while retaining
real negatives. It is not a complete factorial interaction analysis.

The proposal keeps three seeds, initialization, loss, 120 updates, 480 primary
real presentations, 60 auxiliary slots and fixed final checkpoints. Total real
presentations rise from 480 to 510 by design; this must not be hidden as equal
total real exposure. No new rendering or annotation is needed.

This proposal is not implemented, frozen or executed. The current scene-only
auxiliary loader needs an explicit mixed-source input contract. Before execution,
verify training-only membership, negative-mask kind, source roots, original target
provenance, ignored ID 255, cadence, optimizer budget and RNG policy. Use a fresh
matched control or demonstrate legacy-path equivalence before reusing contrasts.
Fewer false positives alone would not qualify utility: every original improvement
and nonregression gate remains required. Wrapped-load coverage remains a separate
unresolved issue. No automatic similar rendering or retraining follows this report.

## Evidence binding

Diagnosis profile SHA256:
`96cf6d10e301abdc12fa587f6d30982a836342e4b64d0d83f6580ece72ef8d04`.
Source utility report SHA256:
`60638e4a25ddda3df5006ce3462fcdec58d97621fe276b02b7b614ae3a26b1b4`.
Reference profile SHA256:
`37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8`.

Full local evidence is under the configured workspace at
`reports/goods_diversity_failure_diagnosis_20261105`, including every frame's
paired transitions, source hashes, seed-vote maps, diagnostic rectangles,
boundary sensitivity, negative-view audit and proposed event schedule.
The source experiment remains closed; retained models, approved labels and the
locked test remain unchanged.

The proposal above records the diagnosis before execution. The subsequent
[real-negative replay diagnostic](real-negative-replay-results.md) implemented
the explicit mixed-source contract and completed a separate three-seed test.
It improved background rejection but harmed dark-goods recall and failed the
complete utility criteria. Original diagnosis artifacts and counts are preserved.
