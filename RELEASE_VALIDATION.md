# Release validation — 1.2.0

This repository revision was built from the uploaded GeoMapBench source archive.

Verified before packaging:

- all Python package and test modules compile;
- the complete offline test suite passes (`9 passed`);
- metric-distance conversion covers metres, kilometres, statute miles, and nautical miles;
- population-density generation balances nine World Bank reference years;
- topological/directional examples declare and render polygon regions;
- high-dynamic-range SpaceNet TIFFs are converted to visible RGB PNGs;
- the existing SpaceNet graph and route task folders can be upgraded through a staged, no-redownload migration;
- the previous eight benchmark-quality corrections remain covered by their tests and validators.

Live external-data builds were not executed in the packaging environment. The included Colab regenerates the three lightweight public-data tasks, upgrades the two existing SpaceNet folders, validates staging copies, and publishes them atomically to Drive.
