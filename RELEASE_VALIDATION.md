# Release validation — 1.1.0

This repository revision was built from the uploaded GeoMapBench source archive.

Verified before packaging:

- all Python package and test modules compile;
- the complete test suite passes (`5 passed`);
- coordinate transformations cover six reversible modes;
- OpenEarthMap RGB labels are converted to validated class-index masks;
- flat and nested MapText sequence layouts preserve multiword groups;
- the ready-to-run Colab contains build commands only for the eight revised leaves;
- the final ZIP was extracted into a clean directory and tested again.

The live external-data builds were not executed in the packaging environment. The Colab performs those downloads/API calls, validates each local task before publishing it to Drive, then runs repository-wide validation.
