"""The console API has two descriptions of itself, and nothing used to
check they agreed.

`lib/api-spec/openapi.yaml` is hand-written and is the input orval turns
into the frontend's typed hooks and Zod schemas (lib/api-client-react,
lib/api-zod). `geocadastra/api/console.py` is the code that actually
answers those requests. They were written at different times, in different
languages, by different people, and the only thing keeping them in step was
that someone had compared them by hand.

That is the same shape of gap that let the whole translation layer be
unreachable in production for weeks: a contract everyone believed, and no
test asserting it. These tests close it -- the spec is now executable.

Deliberately NOT a general OpenAPI validator. The spec uses OpenAPI 3.0's
`nullable:` dialect, which JSON Schema validators do not read (a `jsonschema`
run would silently accept `null` for a non-nullable field, or reject a
legitimately nullable one), so the checker below implements exactly the
constructs this spec uses and fails loudly on anything it has not been
taught.
"""
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from geocadastra.api.main import app, get_db_config, get_session
from geocadastra.jobs.orchestrator import app as celery_app
from geocadastra.jobs.orchestrator import ingest_synthetic_ward
from geocadastra.synth.generator import WardParams, generate_ward
from geocadastra.tests.conftest import TEST_DB_URL, TEST_SCHEMA

pytestmark = pytest.mark.slow

SPEC_PATH = Path(__file__).resolve().parents[2] / "lib" / "api-spec" / "openapi.yaml"

# Paths in the spec that this backend does NOT own. /healthz and the
# sentinel surface are served by artifacts/api-server (Express + its own
# store and Gemini client); asserting Python serves them would be wrong.
NOT_OURS = ("/healthz", "/sentinel")


@pytest.fixture(scope="module")
def spec():
    if not SPEC_PATH.exists():  # pragma: no cover - layout guard
        pytest.skip(f"spec not found at {SPEC_PATH}")
    return yaml.safe_load(SPEC_PATH.read_text())


@pytest.fixture(autouse=True)
def _eager_celery():
    celery_app.conf.task_always_eager = True
    yield
    celery_app.conf.task_always_eager = False


@pytest.fixture()
def client(committed_session):
    app.dependency_overrides[get_session] = lambda: committed_session
    app.dependency_overrides[get_db_config] = lambda: (TEST_DB_URL, TEST_SCHEMA)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def ward(client, committed_session):
    w = generate_ward(
        params=WardParams(width=50, height=35, gsd=1.0, n_arterial_h=1, n_arterial_v=0), seed=2)
    job = ingest_synthetic_ward(committed_session, w, seed=2)
    committed_session.commit()
    client.post(f"/wards/{job.id}/run")
    return job


# ----------------------------------------------------------- the checker --


def _resolve(schema, spec):
    while "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/components/schemas/"), f"unsupported $ref form: {ref}"
        schema = spec["components"]["schemas"][ref.rsplit("/", 1)[1]]
    return schema


def _check(value, schema, spec, path="$"):
    """Return a list of human-readable mismatches. Empty means conforming."""
    schema = _resolve(schema, spec)
    problems = []

    for key in ("oneOf", "anyOf"):
        if key in schema:
            if not any(not _check(value, s, spec, path) for s in schema[key]):
                problems.append(f"{path}: matches no branch of {key}")
            return problems
    if "allOf" in schema:
        for sub in schema["allOf"]:
            problems += _check(value, sub, spec, path)
        return problems

    declared = schema.get("type")
    if value is None:
        # This spec spells optional-null two ways: `nullable: true` on the
        # field (3.0 style) and a `{type: 'null'}` branch inside a oneOf
        # (3.1 style). Both appear, so both are accepted.
        if not (schema.get("nullable", False) or declared == "null"):
            problems.append(f"{path}: null but not declared nullable")
        return problems
    if declared == "null":
        problems.append(f"{path}: expected null, got {type(value).__name__}")
        return problems
    if declared is None:
        return problems  # untyped: nothing to assert

    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        # bool is a subclass of int in Python; a JSON boolean is not a number
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    }
    assert declared in checks, f"checker does not know type {declared!r} (at {path})"
    if not checks[declared](value):
        problems.append(f"{path}: expected {declared}, got {type(value).__name__}")
        return problems

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in enum {schema['enum']}")

    if declared == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                problems.append(f"{path}: missing required property {name!r}")
        for name, sub in properties.items():
            if name in value:
                problems += _check(value[name], sub, spec, f"{path}.{name}")
    elif declared == "array" and "items" in schema:
        for i, item in enumerate(value):
            problems += _check(item, schema["items"], spec, f"{path}[{i}]")

    return problems


def _response_schema(spec, path, method="get", status="200"):
    operation = spec["paths"][path][method]
    return operation["responses"][status]["content"]["application/json"]["schema"]


# ------------------------------------------------------------- the tests --


def test_the_checker_rejects_what_it_should(spec):
    """A conformance test that never fails proves nothing, so check the
    checker: each of these must be reported."""
    schema = {"type": "object", "required": ["a"],
              "properties": {"a": {"type": "string"}, "b": {"type": "number", "nullable": True}}}
    assert _check({"a": "x", "b": 1.0}, schema, spec) == []
    assert _check({"a": "x", "b": None}, schema, spec) == []
    # the 3.1-style spelling this spec also uses, inside a oneOf
    null_branch = {"oneOf": [{"type": "string"}, {"type": "null"}]}
    assert _check(None, null_branch, spec) == []
    assert _check("x", null_branch, spec) == []
    assert _check(5, null_branch, spec), "oneOf with no matching branch not caught"
    assert _check({"b": 1.0}, schema, spec), "missing required property not caught"
    assert _check({"a": 5}, schema, spec), "wrong type not caught"
    assert _check({"a": "x", "b": "no"}, schema, spec), "wrong nested type not caught"
    assert _check({"a": None}, schema, spec), "null in non-nullable field not caught"
    assert _check([], schema, spec), "wrong container type not caught"


def test_every_spec_path_this_backend_owns_is_actually_served(client, spec):
    """The failure that started all this: a path in the spec (so in the
    generated client, so called by the frontend) that nothing here answers."""
    served = set(app.openapi()["paths"])
    missing = [p for p in spec["paths"] if not p.startswith(NOT_OURS) and p not in served]
    assert not missing, f"declared in openapi.yaml but not served by FastAPI: {missing}"


def test_declared_methods_match(client, spec):
    served = app.openapi()["paths"]
    mismatched = []
    for path, operations in spec["paths"].items():
        if path.startswith(NOT_OURS):
            continue
        for method in operations:
            if method in ("get", "post", "patch", "put", "delete") and method not in served.get(path, {}):
                mismatched.append(f"{method.upper()} {path}")
    assert not mismatched, f"declared in openapi.yaml but not served: {mismatched}"


# Served by this backend but deliberately absent from the published
# contract. The frontend reaches these through hand-written fetch helpers
# (utils/backend.ts, operator/api.ts, operator/workspace.ts) rather than
# generated hooks, so they have types on the client but no shared schema.
# Listed explicitly so that ADDING a route without deciding which side of
# that line it belongs on fails here instead of going unnoticed.
UNCONTRACTED_PREFIXES = ("/wards", "/workspace", "/advisory")


def test_the_uncontracted_surface_does_not_grow_silently(client, spec):
    served = set(app.openapi()["paths"])
    contracted = set(spec["paths"])
    uncovered = sorted(p for p in served if p not in contracted)
    unexpected = [p for p in uncovered if not p.startswith(UNCONTRACTED_PREFIXES)]
    assert not unexpected, (
        "these routes are served but described nowhere -- either add them to "
        f"lib/api-spec/openapi.yaml or to UNCONTRACTED_PREFIXES: {unexpected}")


@pytest.mark.parametrize("path", ["/dashboard", "/regions", "/processing/runs", "/changes"])
def test_collection_responses_conform_to_the_spec(client, ward, spec, path):
    body = client.get(path).json()
    problems = _check(body, _response_schema(spec, path), spec)
    assert not problems, f"{path} does not match its declared schema: {problems}"


def test_parcels_conform_to_the_spec(client, ward, spec):
    body = client.get(f"/parcels?regionId=ward-{ward.id}").json()
    assert body, "ward produced no parcels; this assertion would be vacuous"
    problems = _check(body, _response_schema(spec, "/parcels"), spec)
    assert not problems, f"/parcels does not match its declared schema: {problems}"


def test_single_parcel_conforms_to_the_spec(client, ward, spec):
    first = client.get(f"/parcels?regionId=ward-{ward.id}").json()[0]
    body = client.get(f"/parcels/{first['id']}").json()
    problems = _check(body, _response_schema(spec, "/parcels/{id}"), spec)
    assert not problems, f"/parcels/{{id}} does not match its declared schema: {problems}"
