# Security and data issues

The repository includes the processed L=3 covariate backbone described in
`DATA_NOTICE.md`. No other Compustat extract, raw download, access credential,
or generated firm-level result belongs in the repository.

Please report suspected disclosure or credential issues privately to the
maintainer listed in `CITATION.cff`. Do not attach firm-level records or access
credentials to a public issue.

The repository scan is run with:

```powershell
python tools/audit_public_release.py
```

It checks the authorized backbone hash and scans for additional data files,
archives, office documents, local absolute paths, symlinks, and likely
credential assignments. The file list should also be reviewed before each
release.
