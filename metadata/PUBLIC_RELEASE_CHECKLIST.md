# Public release checklist — v1.1.0

- [x] Code extracted (cyp-atlas/src/, hdi-gnn/src/)
- [x] Temporary files removed (__pycache__, .ipynb_checkpoints)
- [x] Private files removed (no credentials, no local paths, no user data)
- [x] Secret scan completed (no api_key/password/token in tree)
- [x] Data provenance recorded (metadata/SOURCE_PROVENANCE.md)
- [x] Frozen datasets generated (data/frozen/*.csv)
- [x] SHA-256 checksums generated (metadata/CHECKSUMS.sha256)
- [x] Data dictionary available (metadata/DATA_DICTIONARY.md)
- [x] Model artifacts documented (models/README.md; checkpoint EXCLUDED, regenerable)
- [ ] Model checksums generated (checkpoint is external to git — generate at release upload)
- [x] Random seeds recorded (metadata/REPRODUCIBILITY_MANIFEST.json)
- [x] Environment frozen (requirements.txt, environment.yml)
- [x] Configuration frozen (not applicable; CLI args + default seeds)
- [x] README completed (reproducibility quick-start + v1.1.0 section)
- [x] CITATION.cff present (v1.1.0; ORCID/DOI TBD before Zenodo archival)
- [x] LICENSE verified (MIT for code; third-party data keeps own terms)
- [x] CHANGELOG completed (v1.1.0 entry)
- [x] Reproducibility manifest created
- [ ] Tests passed (python -m compileall cyp-atlas/src hdi-gnn/src)
- [ ] Clean-room reproduction completed (metadata/REPRODUCIBILITY_TEST.md)
- [ ] Git history inspected (git log --all --source -S'password|token|api_key')
- [ ] Git tag created (git tag v1.1.0)
- [ ] GitHub Release created (requires repo owner credentials)
- [ ] Zenodo archival completed (requires owner authorization)
- [ ] DOI verified
- [ ] Final public repository audited

The unchecked items require a shell on your machine with git credentials and
network access to GitHub; see the release procedure in the deposition report.
