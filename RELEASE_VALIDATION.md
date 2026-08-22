# Release validation — 1.3.0


## Bloom-balanced overlay checks

After the base 23-leaf release passes the existing checks, run:

```bash
geomapbench-data bloomify --root "$OUT"
geomapbench-data validate --root "$OUT" --require-all
geomapbench-data bloom-audit --root "$OUT"
```

Expected Bloom audit:

- 23 leaves;
- 2,300 records;
- every five-level leaf has exactly 20 examples per supported level;
- every six-level leaf has exactly `17,17,17,17,16,16` examples in the declared level order;
- every record contains `bloom.revision = 2026-08-bloom-v1`;
- every record contains `target.bloom_answer`;
- every `evaluation.target_field` is `target.bloom_answer`;
- original target fields and all referenced assets remain present.

The converter stores pre-Bloom `data.jsonl` and `manifest.json` under each leaf's `.pre_bloom/` directory.

This repository revision was built from the uploaded GeoMapBench source archive.

Verified before packaging:

- all Python package and test modules compile;
- the complete offline test suite passes (`12 passed`);
- metric-distance conversion covers metres, kilometres, statute miles, and nautical miles;
- population-density generation balances nine World Bank reference years;
- topological/directional examples declare and render polygon regions;
- high-dynamic-range SpaceNet TIFFs are converted to visible RGB PNGs;
- the existing SpaceNet graph and route task folders can be upgraded through a staged, no-redownload migration;
- the previous eight benchmark-quality corrections remain covered by their tests and validators.

Live external-data builds were not executed in the packaging environment. The Bloom-update Colab operates only on an already-built 23-leaf release: it validates the base dataset, applies the deterministic Bloom overlay in place, validates the result, and writes a Bloom audit without redownloading any upstream data.
