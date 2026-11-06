# Opaque diversity utility result

The completed three-seed comparison **failed** the frozen acceptance rules.
The approved 35-image collection passed technical and appearance admission, but
its training contribution did not qualify. The single-candidate campaign is
closed. No model was promoted, no references changed and no locked test ran.

| Three-seed mean | Fresh real-only | With synthetic | Change |
| --- | ---: | ---: | ---: |
| Pixel-weighted goods IoU | 50.16% | 46.69% | -3.47 pp |
| Precision | 54.20% | 49.68% | -4.52 pp |
| Recall | 87.08% | 88.59% | +1.51 pp |
| Background false-positive rate | 8.05% | 9.93% | +1.89 pp |
| Dark-goods miss rate | 36.48% | 27.16% | -9.32 pp |
| Frame-average goods IoU | 42.60% | 40.81% | -1.79 pp |

Rates are pixel-weighted within each seed and then averaged across three seeds.
Frame-average IoU is reported separately. Recall and dark-goods nonregression
passed; six other requirements failed. The 27.16% dark miss rate remains above
the frozen 18.67% ceiling. The required overall IoU was at least 53.16%.
No seed improved overall IoU; the rule required at least two. Three frames
improved, eight regressed and one was unchanged.

The largest mean per-frame IoU regressions were development frame 000188
(-13.74 pp), 000282 (-10.49 pp) and 000659 (-2.60 pp). Prior visual descriptions
identify blurred shelf goods in 000188 and wrapped-load/shelf surfaces in 000282;
these are context notes, not independent object-class annotations. The report
shows fixed seed 29 for every development image and retains all 72 native
prediction recounts, including improvements and regressions.

## Protocol and verification

Both arms used seeds 17, 29 and 43, identical generic pretrained initialization,
120 updates, 480 primary real and 60 auxiliary presentations per seed, and the
fixed final checkpoint at epoch 15. The control used 60 real replays. The
candidate visited each of 30 loaded donors once and each of five empty donors
six times. Both had 30 mixed and 30 background-only auxiliary presentations.
All 32 primary real frames were presented 15 times in every run. Recorded donor
visits reproduce the frozen schedules exactly. Auxiliary class-mean loss makes
pixel totals different from gradient weights; ignored real targets stay ignored.

Fourteen checks passed on the exact isolated source/configuration snapshot.
Every saved checkpoint was strictly reloaded. All 72 native development masks
were independently recounted. The fresh control reproduced all 36 historical
development predictions exactly. All 1,071 protected report/model artifacts
were verified unchanged after execution and reporting.

Training used 640x360 RGB and nearest-neighbour targets, with logits evaluated at
1280x720. Precision was float32 on the RTX 5070 Laptop GPU with four CPU threads.
Peak torch CUDA allocation was 1,415,031,296 bytes (1.32 GiB); this excludes
reserved and other process/device memory. The six recorded seed-run intervals
sum to 397.36 seconds. Both complete arms plus comparison took 614.40 seconds,
excluding preparation and report generation. This is not accepted-data throughput.

## Source binding and interpretation

The [aggregate evidence](goods-diversity-results.json) binds references, collection,
code, protocol, initialization, all six checkpoints, metrics and gate outcomes.

| Binding | SHA256 |
| --- | --- |
| Reference profile | `37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8` |
| Admitted collection | `818ca42f7c3518f8cabb8a8da187b2a511584519c866bdfca64e5a15438a502b` |
| Executed code snapshot | `9831d67f3f6331ecfde8868ea9836f54fa5bdce2314d0861e3aa7645106ed5c0` |
| Frozen protocol | `d33cb52e9839697fb8a692f23ca2fe30c6a8dcd09629ecf62b8e62c0aa8a137e` |
| Comparison profile | `2a573a2f2faf7bd282e3e4176608bca429edf2d20c226772426a7d9030ae8fe9` |
| Verified report profile | `60638e4a25ddda3df5006ce3462fcdec58d97621fe276b02b7b614ae3a26b1b4` |

Full local evidence is under the configured workspace at
`reports/goods_diversity_comparison_20261105`.

The result supports a failure of this collection under this protocol, not a
proof that every synthetic approach fails. Asset, camera and arrangement factors
changed together; their individual effects are not isolated. Related frames,
repeated development use and human-reviewed model-assisted references limit
generalization claims. The known dark-coverage gap is not a proven explanation
of the utility regression. Preserve the approved labels and retained model;
investigate the observed false-positive and frame regressions before proposing
another bounded experiment. More similar rendering or reannotation is not an
automatic next step. The earlier layout campaign remains closed separately.
