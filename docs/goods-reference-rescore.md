# Approved development-reference rescore

Explicit user acceptance qualified a new version of development image 000282:
3,235 ignored pixels on a visible carton became goods, and 755 ignored pixels
on the adjacent post became background. Existing labelled pixels, all 32 training
targets and the other 11 development targets are unchanged. References remain
human-reviewed model-assisted annotations, not independent ground truth.

All 108 saved development predictions across three arms and seeds 17, 29 and 43
were rescored. No training, inference, checkpoint selection, model promotion or
locked-test evaluation occurred. Original references and reports are preserved.

| Three-seed mean | Real-only | Synthetic negatives | Real negatives |
| --- | ---: | ---: | ---: |
| Goods IoU | 50.24% | 46.77% | 49.34% |
| Precision | 54.28% | 49.77% | 54.68% |
| Recall | 87.13% | 88.63% | 84.14% |
| Background false-positive rate | 8.05% | 9.94% | 7.88% |
| Dark-goods miss rate | 36.48% | 27.16% | 45.83% |

Both synthetic candidates still fail the original utility requirements. Fixed
thresholds remain unchanged; control-relative requirements also use the rescored
control. This posthoc development review does not establish independent
generalization or repair missed wrapped goods.

The new reference profile SHA256 is
`c62285cc61d42c2f91ac557734382df78fab277d2563a6868e4fbed6bd13c14c`.
Its parent training/evaluation reference profile is
`37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8`.
Full evidence is in workspace report `goods_carton_reference_rescore_20261105`.

## Coverage qualification

A training-only audit inspected all 32 reviewed training images and nine local
asset catalogs. Real goods contain 336,111 dark pixels out of 3,208,811 labelled
goods pixels (10.47%); the accepted synthetic collection contains 144,210 out of
2,821,793 (5.11%). Darkness uses luminance below 50. This is a coverage diagnostic,
not a causal explanation or a utility acceptance criterion.

Training images contain wrapped loads, access stairs, lifting equipment and
close rack occlusions beyond the qualified opaque carton/tyre assets. Current
assets allow partial changes to grouping and lighting, but do not establish
coverage of those missing appearances. No further batch is admitted or experiment
started on the strength of this audit. Further training requires a meaningfully
different, qualified candidate and a separately frozen protocol.
