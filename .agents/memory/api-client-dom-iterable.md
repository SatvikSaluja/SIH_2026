---
name: Generated client DOM iterable requirement
description: Workspace compiler detail needed by generated fetch client code.
---

Generated API client code calls `Headers.entries()`, so the client library TypeScript `lib` configuration must include `dom.iterable` alongside `dom`.

**Why:** The generated client compiles successfully but the workspace library typecheck fails without iterable DOM declarations.

**How to apply:** Preserve `dom.iterable` when regenerating or refactoring the shared API client package.