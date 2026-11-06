# Broader opaque goods collection

The collection contains 30 loaded images and five distinct equipment-only images,
in five jobs of seven captures. Each camera has six loaded arrangements and one
shared empty counterpart with identical equipment. This avoids duplicate empty
donors. All images are native 1280x720 exports. The user explicitly accepted
the exact 35-image RGB/label collection. All five admissions and the verified
collection input are complete. The subsequent [controlled comparison](goods-diversity-results.md)
completed and failed the frozen utility requirements; no model was promoted.

Admitted collection profile SHA256:
`818ca42f7c3518f8cabb8a8da187b2a511584519c866bdfca64e5a15438a502b`.
The original review snapshot remains unchanged; approval is recorded separately.

The six families are regular cartons, wide cartons, tall packages, explicitly
authored dented cartons, tyres and mixed stock. Existing registered assets and
the accepted package-scout library are reused. An upright rotation or authored
dent is not another scanned source. Five cameras in one warehouse are correlated
views, not 30 independently acquired environments.

The preceding eight-image scout was explicitly accepted by the user and admitted
under profile SHA256
`de8694728948f71e9673dfdc525442f6c8b70ef6acca667a37800b5d37802d80`.
That approval does not extend to these 35 newly rendered captures.

## Technical evidence

All five jobs passed native-array checks and independent geometry qualification.
All 8,960 sampled instance hits agree; maximum per-view axial-depth p95 residual
is 0.000060 metres. All 32,256,000 native target pixels were checked against
visible instance classes. The intended goods families are visibly present; mixed
views contain both tyres and cartons. All five empty images have zero goods
pixels. All 35 RGB payloads are distinct. Cargo placement, support containment
and declared contacts were checked separately from appearance.

A frozen source copy prevents concurrent repository edits from changing the
rendered batch. The code-snapshot profile SHA256 is
`e587248a24b1199cbb1d8e8e1513238e63c89883fa5a3f63942ee58c371b5657`.
The final review profile SHA256 is
`9db62e7eb24bd0f4b419a64e7bb62ce02dcbc7a789503fd456e5cf9f8b5b2a70`.
Its five member dataset and geometry digests are retained in the review profile.
The 1,043 protected prior report files, including six saved checkpoints, were
verified unchanged. Real references remain bound to
`37ab46daec17a248cc5dc42016113159d42798e64fafbfcafe9ea88a71a36be8`.

## Training-only coverage

| Measurement | Real training references | Synthetic loaded views |
| --- | ---: | ---: |
| Images | 32 | 30 |
| Median visible-goods area | 10.22% | 8.73% |
| Maximum visible-goods area | 31.48% | 21.64% |
| Dark fraction of goods pixels | 10.47% | 5.11% |

Area uses the whole native image as denominator in both columns. Dark means
Pillow RGB-to-L display luminance below 50/255. Ignored real pixels remain ignored
and never count as goods. These are pixel-coverage figures, not training-gradient
weights, accuracy results or sensor-calibrated measurements. The overall dark
coverage gap remains; tyre-only views contain more dark goods than the carton
families. Histogram agreement alone would not establish usefulness.

## Utility decision

The separate comparison used fresh controls and candidates with seeds 17, 29 and
43, 120 updates, 480 primary real and 60 auxiliary presentations per seed, and
fixed final checkpoints. It visited every loaded donor once and each empty donor
six times. Mean goods IoU fell from 50.16% to 46.69%. Recall and dark misses
improved, but precision and false detections worsened. Six of eight frozen
requirements failed. The single-candidate campaign is closed.

The report retains all improvement/nonregression gates and every seed/frame
regression. The earlier layout campaign remains closed separately.
Transparent wrapping, blur, material response, independent environment coverage
and physical sensor fidelity remain limited. Exact-batch acceptance and a
technical pass do not establish synthetic utility or data-complete status.

Full local evidence is under the configured workspace at
`reports/goods_diversity_collection_20261104`.
