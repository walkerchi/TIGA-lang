# Security policy

## Reporting a vulnerability

Do not open a public issue containing a security report. Send private reports to
**walkerchi** at [walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com),
with the subject prefix `[Tiga security]`.

When enabled, use **Security → Report a vulnerability** in the
[official repository](https://github.com/walkerchi/TIGA-lang/security).
If unavailable, use the email address above without posting exploit details publicly.
Include:

- the affected version (`python -m tiga` output) and platform,
- a minimal reproduction or a description of the impact,
- whether the report may be credited in the changelog once fixed.

Tiga is pre-release alpha software with no stable supported version or promised
response-time SLA. Coordinate disclosure with the maintainer; remove private
data, credentials and identifying machine information from reports.

## Scope

Tiga executes native code. Python UDFs, provider plugins and generated artifacts
are executable code, not sandboxed inputs. Do not expose compilation as an
unauthenticated service or run untrusted programs/libraries. Experimental TCP
transport is for trusted networks; it does not establish encryption or authentication.

Memory-safety defects and malformed-file handling are relevant security reports.
Snapshot and visualization loaders are not certified safe for adversarial files;
use external isolation where needed. A compiler verifier is not a security sandbox.
