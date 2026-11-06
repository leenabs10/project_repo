# Current status

## Pipeline

Ingestion, profiling, controlled generation, geometry/label checks, source-bound
admission, saved reference review and matched goods evaluation are implemented.
The maintained renderer provides RGB, axial depth, validity, semantic and instance
outputs. Qualified simulated LiDAR and calibrated physical sensor fidelity remain open.

## Selected references

32 training and 12 development images. The latest qualified evaluation revision
includes the explicitly accepted carton/post correction in development image
000282. All training targets and the other 11 development targets are unchanged.
Profile SHA256: `c62285cc61d42c2f91ac557734382df78fab277d2563a6868e4fbed6bd13c14c`.
Historical training, admissions and original results remain bound to profile
`37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8`;
the new evaluation revision does not silently rebind those artifacts.

## Latest utility decision

The [approved-reference rescore](goods-reference-rescore.md) reuses 108 saved
development predictions without training or inference. Corrected-reference IoU
is 50.24% for real-only, 46.77% for synthetic goods with synthetic negatives,
and 49.34% for synthetic goods with real negatives. Both candidates still fail
the original utility requirements. The following historical scores retain their
original reference binding.

The [real-negative replay diagnostic](real-negative-replay-results.md) completed
on 2026-11-05. Keeping the accepted synthetic positives while restoring real
training-negative replay reduced background false positives from 9.93% to 7.87%
and raised precision from 49.68% to 54.60%, relative to the prior synthetic arm.
However, recall fell to 84.08% and dark-goods misses rose to 45.83%. Mean IoU is
49.25%, below the fresh control's 50.16%. Six of eight unchanged utility gates
fail; this one-candidate diagnostic is closed and no model was promoted.
The fresh control reproduces 36 predictions, 1,080 model tensors and three
histories exactly. All 396 saved training/development masks in the three-arm
audit were checked. The retained model and annotations remain unchanged.

The [opaque diversity comparison](goods-diversity-results.md) completed on
2026-11-05 and failed six of eight frozen requirements. Mean goods IoU fell from
50.16% to 46.69%, precision fell from 54.20% to 49.68%, and background false
detections rose from 8.05% to 9.93%. Recall and dark-goods misses improved, but
the dark-miss ceiling still failed. All three seeds lost IoU. The fresh control
reproduced its 36 historical predictions, and all 72 new development masks were
independently recounted. This separate single-candidate campaign is closed;
no model was promoted and the locked test remains unevaluated.

The subsequent [saved-prediction diagnosis](goods-diversity-diagnosis.md) identifies
new false detections on real guards, handling equipment, empty pallets and access
stairs. Most added false-positive pixels lie outside a ten-pixel boundary band.
The five synthetic empty views share one equipment arrangement. The subsequent
mixed-source diagnostic is now implemented, tested and complete. Its background
improvement and dark-goods regression show that restoring real negatives alone
does not qualify the current synthetic-positive representation.

### Earlier layout campaign

The two-candidate campaign is closed. Neither candidate passed the complete frozen
requirements. No model was promoted; the retained reporting model is unchanged.

| Metric | Fresh real-only | Dense-layout synthetic |
| --- | ---: | ---: |
| Goods IoU | 50.16% | 50.96% |
| Precision | 54.20% | 57.04% |
| Recall | 87.08% | 83.59% |

The fresh control reproduced its earlier 36 native predictions exactly. All 72
development predictions were recounted. The earlier grouped-cargo candidate also
failed. Similar scores do not establish realism or overall synthetic usefulness.

## Current quality work

Separate package-asset scouts preserve their original qualification states.
Geometry and appearance qualification apply only to each exact
batch; no acceptance is inferred from relocation or collection assembly. This
work is not a third candidate in the closed campaign. Original sources, semantic
decisions and limitations remain in the archived qualification records and the
external workspace evidence.

The user explicitly accepted the eight-view package scout; its separate admission
is recorded. The subsequent [broader goods collection](goods-diversity.md) contains
30 loaded views and five shared empty controls. All 8,960 sampled instance hits
agree and native labels pass. The user accepted all 35 images, and source-bound
admission and collection assembly are complete. Its controlled training comparison
has now completed and failed utility qualification.
Dark goods comprise 5.11% of synthetic goods
pixels versus 10.47% in the real training references, so coverage gaps remain.

## Remaining gates

Saved-model handover verification on 2026-11-06 strictly reloaded all six
real-only/mixed-source checkpoints. All 72 native development evaluations
reproduced their recorded confusion counts, and the fixed seed-17 demonstration
models reproduced 88 saved training/development masks byte for byte. The current
software suite passed 295 tests with one skip and 238 passing subtests. These
checks verify reproduction, not synthetic utility or a fresh generation job.

- Qualify representative asset/material and scene coverage before another campaign.
- Declare a separate supported hypothesis, budget and stopping rule for any new trial.
- Establish an acceptable complete utility result, not an isolated precision gain.
- Freeze selection and protocol before independently qualified held-out evaluation.
- Package accepted data, configurations, quality/coverage evidence and limitations.
- Resolve the outstanding same-scene simulated-LiDAR requirement for full multimodal scope.

The locked test remains unevaluated. Related frames, repeated development use
and reviewed model-assisted references limit generalization claims.

Use `mvp.cmd reports` for exact versions and full evidence. Original result
summaries are preserved in `archive/docs/` without changing their measurements.
