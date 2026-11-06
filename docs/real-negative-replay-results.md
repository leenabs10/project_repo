# Real-negative replay diagnostic

One three-seed diagnostic tested whether the source of auxiliary background
examples contributed to the equipment errors. It reduced background false
detections compared with the prior synthetic arm, but reduced goods recall and
failed six of eight unchanged utility requirements. No model was promoted.

| Three-seed mean | Fresh real-only | Previous synthetic | Synthetic positives with real negatives |
| --- | ---: | ---: | ---: |
| Goods IoU | 50.16% | 46.69% | 49.25% |
| Precision | 54.20% | 49.68% | 54.60% |
| Recall | 87.08% | 88.59% | 84.08% |
| Background false-positive rate | 8.05% | 9.93% | 7.87% |
| Dark-goods miss rate | 36.48% | 27.16% | 45.83% |
| Frame-average IoU | 42.60% | 40.81% | 40.53% |

Rates are pixel-weighted within each seed and then averaged. Background false
detections decrease versus the previous synthetic arm in all three seeds.
Only seed 43 improves overall IoU versus the fresh control; two gains were
required. Mean IoU remains below the control and the required 53.16% target.
Recall, both dark-miss requirements and frame-average IoU also fail. The original
18.67% dark-miss ceiling remains in force.

## What the controlled intervention establishes

The exact 30 accepted synthetic-positive events are preserved. The 30 synthetic
negative events are replaced by the original real training-negative schedule,
using frames 000482 and 000880. Initialization, model, normalization, losses,
primary training order, 120 updates, three seeds and final-epoch selection are
fixed. Primary real exposure remains 480 presentations per seed. Total real
exposure changes from 480 in the previous synthetic arm to 510 in this diagnostic;
the fresh real-only control has 540. This is the declared source/content change.

The outcome supports a contribution from background source/content under this
protocol. It does not isolate synthetic equipment variety as the unique cause,
or provide a full factorial interaction analysis. The two real negative examples
were already classified correctly in training. Replaying them improves some
background decisions but does not teach a robust distinction across all real
equipment and goods appearances.

## Native image findings

The five reported real-image problems were inspected against unchanged reviewed
references. The highlighted equipment remains background; the visible wrapped
load in 000282 is already goods. No specific reference correction was established
that would explain the failure.

Rates within the previously declared diagnostic rectangles improve versus the
prior synthetic arm: guard-region false positives decrease from 35.66% to 30.56%,
lifting-platform-region false positives from 23.58% to 19.21%, close-pallet-region
false positives from 32.99% to 30.00%, and stair-region false positives from 12.54%
to 9.70%. These rectangles can include multiple surfaces; rates count only their
existing valid reference pixels and are not object-level accuracy measurements.

Wrapped-load-region recall improves from 65.79% to 73.64%, remaining below the
control's 82.93%. Near-tyre-region recall falls from 65.35% to 39.59%, below the
control's 53.37%. Gains in equipment rejection therefore coexist with substantial
goods losses. All 12 frames and three seeds are retained in the report.

On the same training references, the new arm reaches 92.97% goods IoU with a
0.83% background false-positive rate. Development IoU is 49.25% with a 7.87%
background false-positive rate. This is evidence of a generalization gap, not
proof of one causal mechanism. Reannotation or more repeated views alone is not
supported as a solution.

## Input support and verification

The new opt-in `explicit_mixed_source_sequence` validates the source of each
auxiliary event, training membership, target kind, native categorical mask,
contained source paths and complete update budget. Each image retains its own
RGB/target root, including ignored pixels. Execution records source and payload
hashes and counts real replay as real. Legacy sampling behaviour is preserved.

The full software suite passed 296 tests, with one skip. A 32-test focused run
also passed; these counts overlap. The fresh control exactly reproduces all 36
historical development predictions, 1,080 model tensors and three training
histories. Independent scoring checks all 396 saved masks across the three arms:
108 development and 288 training masks. Confusion counts match exactly; frame
average verification permits only floating summation differences up to 1e-12.
No extra inference was required for this recount.

## Decision and next evidence

This single-candidate diagnostic is closed. Preserve existing labels and the
retained reporting model. The software now supports a controlled mixed-source
input schedule; the dataset has not met the complete utility milestone.

Further work should qualify coverage of real equipment and identifiable goods
together: varied guard, lift, stair and pallet appearances alongside dark goods
and partially occluded loads. Use training-only sources to define any new batch
or reference selection. Transparent wrapping remains unsupported by the current
opaque scene assets. Increasing background pressure alone is not supported as
a complete fix because the measured dark-goods losses are substantial.

A later experiment needs a separately frozen hypothesis, budget and stopping
rule. Do not continue the closed campaigns with score-driven label changes or
threshold tuning. Repeated development use, correlated video frames and
human-reviewed model-assisted targets limit conclusions. The locked test remains
untouched and no independent generalization claim is made.

## Evidence binding

| Artifact | SHA256 |
| --- | --- |
| Reference profile | `37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8` |
| Accepted source collection | `818ca42f7c3518f8cabb8a8da187b2a511584519c866bdfca64e5a15438a502b` |
| Frozen code snapshot | `bfdfcbafa1df1b74ded8c3f4cf09adadbfac7282fce2dcde51e69f2d32eeae27` |
| Frozen diagnostic protocol | `64f6f58114f7915ff0d841a5580189dfb893db7eb1f4e84c755cdc2b890f76b2` |
| Verified result report | `6f3136ce54cf93696b1812324ed347ad4d4981e8f0ffd37de5b94051394c3f96` |

Full evidence is under the configured workspace at
`reports/goods_real_negative_diagnostic_20261105/report_verified`.
