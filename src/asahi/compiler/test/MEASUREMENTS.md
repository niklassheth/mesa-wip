# Archived compiler measurements

Historical CSV/JSON results moved to the workspace artifact directory
`artifacts/mesa-apple9-measurements-20261016/`, with their paths below
`src/asahi/compiler/test/` preserved. This directory is outside the Mesa
repository; the driver and unit tests do not need it.

The artifact `manifest.json` records source revision
`c1786dcc953571834218ad13ec6985e095bedbbb`, original paths, sizes and SHA-256
hashes. Copies were verified before removing the repository copies.
Regeneration and analysis scripts stay here. Historical report links point
at this index; resolve each named result under the artifact directory.

`cs2-pressure/uniform-at-cap.csv` stays beside `rank-next-levers.py`, which
reads it as an input. Generated `cases.json`, `opcodes.json`,
`texture-cases.json` and `expected.json` are not checked-in results and are
still generated/read by their existing test scripts.

## cs2-pressure

| Result path under the artifact directory | SHA-256 |
|---|---|
| `cs2-pressure/compute-deltas.csv` | `fdd9604a2f690808fad4c42813a7d50eeea7a2134dc72a11b7edad3385eda73d` |
| `cs2-pressure/constant-census-results.json` | `381b2f6bd753600ebcac9f28eb2b7f297faecc9e07813757b24f345f11af842e` |
| `cs2-pressure/constant-consumer-census.csv` | `65c944a574990c04d46c5c369964822df5dc65eb9cf3232731757ae4da8c169d` |
| `cs2-pressure/constant-placement-deltas.csv` | `3b8323eb4d78ff703843aeed659fb530d6b129262104270fce811721b02338d4` |
| `cs2-pressure/copy-attribution.csv` | `9a76c9384cd766471c5d5674a11b207391d89e74935f8446eb1439975166141f` |
| `cs2-pressure/copy-summary.json` | `1409c44047b400fc920690a875de78164b6e6abdd3a545c8492f7bf45efa747b` |
| `cs2-pressure/cts/selection.json` | `26a79501439e72dab33cff08705464697c4047e24ef35f27160a85e1c3155b37` |
| `cs2-pressure/licm-offline/benefit-sweep.json` | `d91470d23c8dd024288778e8f3c4c18f14f61bb18bb595ada9a7d81e32bd41cd` |
| `cs2-pressure/licm-offline/budget-sweep.json` | `8679f713a584c1af809419cb74887798ab081b1328ff0a58f41837d266cf9506` |
| `cs2-pressure/licm-offline/loops-summary.json` | `0f7e8c8643a46e7dbefca859c7929ecd17febec0b690d729408bcf6187e46789` |
| `cs2-pressure/licm-offline/on-per-loop.csv` | `c73c428b78a357724caebffe32a79943c800169245ab7a5edda43bf36f078383` |
| `cs2-pressure/licm-offline/restricted-per-loop.csv` | `38cfadb74223e43e3a7c99e1400fbcf84838b72e43681c1d4099ee12051fe324` |
| `cs2-pressure/next-levers-at-cap.csv` | `dd75314ad0b64273b895f84afbf72ccfdbc65982b358a5f49e35f68970c1b8a1` |
| `cs2-pressure/next-levers-census.json` | `8c8966bd710918fe220fa83133d6258e58b274765ae0c1c153e7526e975ff81e` |
| `cs2-pressure/phi-edge-liveness-after-placement.csv` | `8b80453c1d0d765244eb5ce481a2324c9409c748f0a5a2f3d07122e08ea2cec4` |
| `cs2-pressure/phi-edge-liveness-after-uniforms.json` | `ab8e1d6eddc92f82bcf1eb8c5d58e672ad42be67f4eea755a93c09e0a74fa58f` |
| `cs2-pressure/phi-edge-liveness-summary.json` | `636e5e3897f00daf66a99340a0df5355a2dd9c2727c1233c9c7bed939a499652` |
| `cs2-pressure/phi-edge-liveness.csv` | `4e348cefacde057c0fed44c093426f9da64f8c07f4b3829f02fd6f13b5874617` |
| `cs2-pressure/spill-latency-at-cap.csv` | `b1a567ab1cba51324e129df4d1fea9cb7259fe411608c43674f8004fbe9377b6` |
| `cs2-pressure/spill-latency-results.json` | `54b3583161fff87c97768d5bd9fc16ee67e8fdfa860280b1a525883b2e784c20` |
| `cs2-pressure/uniform-replay-verification.json` | `fea36449bb61f2ec0934b81649681d78c47e148a5314a1fe8688c5d8aff1c8c5` |
| `cs2-pressure/uniform-source-probes.json` | `43489107b36012f8a0c494940aea2817faab34bab954bbf8383a8cb028ab8df7` |
| `cs2-pressure/uniform-source-results.json` | `cedc9ee8036dac3b8a1a5b8160e60a61656924da88adf8c0a03588da0001ab08` |
| `cs2-pressure/vertex-regressors.csv` | `278bf16ec9c88389e97004312a4d99bfad35220eaefc56a6cd82e3bca9433392` |
