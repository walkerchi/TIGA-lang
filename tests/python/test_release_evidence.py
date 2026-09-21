"""Publication must reuse a successful tagged build and the approved wheel."""
import hashlib

import pytest

from tools.verify_release_evidence import verify_evidence


@pytest.fixture
def evidence(tmp_path):
    wheel = tmp_path / 'tiga_lang-0.1.0-cp312-cp312-manylinux_2_38_x86_64.whl'
    wheel.write_bytes(b'GPU-tested wheel')
    repository = 'walkerchi/TIGA-lang'
    commit = 'a' * 40
    run = dict(status='completed', conclusion='success', event='push',
               path='.github/workflows/release.yml', head_branch='v0.1.0',
               head_sha=commit, repository={'full_name': repository},
               head_repository={'full_name': repository})
    options = dict(commit=commit, repository=repository, tag='v0.1.0',
                   gpu_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    return tmp_path, run, options


def test_matching_completed_build_and_wheel(evidence):
    directory, run, options = evidence
    assert verify_evidence(directory, run, **options).endswith('.whl')
    run['event'] = 'workflow_dispatch'
    options['gpu_wheel_sha256'] = options['gpu_wheel_sha256'].upper()
    assert verify_evidence(directory, run, **options).endswith('.whl')


@pytest.mark.parametrize(('field', 'value'), [
    ('status', 'queued'), ('conclusion', 'failure'), ('conclusion', 'cancelled'),
    ('path', '.github/workflows/compiler-ci.yml'), ('head_branch', 'main'),
    ('head_sha', 'b' * 40), ('repository', {'full_name': 'fork/TIGA-lang'}),
    ('head_repository', {'full_name': 'fork/TIGA-lang'}), ('event', 'pull_request'),
])
def test_reject_wrong_or_incomplete_build(evidence, field, value):
    directory, run, options = evidence
    run[field] = value
    with pytest.raises(ValueError):
        verify_evidence(directory, run, **options)


@pytest.mark.parametrize('digest', ['', '0' * 64, 'x' * 64, 'f' * 63])
def test_reject_missing_invalid_or_different_digest(evidence, digest):
    directory, run, options = evidence
    options['gpu_wheel_sha256'] = digest
    with pytest.raises(ValueError):
        verify_evidence(directory, run, **options)


def test_reject_wheel_changed_after_local_gpu_validation(evidence):
    directory, run, options = evidence
    next(directory.glob('*.whl')).write_bytes(b'different wheel')
    with pytest.raises(ValueError, match='differs'):
        verify_evidence(directory, run, **options)


@pytest.mark.parametrize('count', [0, 2])
def test_require_exactly_one_supported_gpu_wheel(evidence, count):
    directory, run, options = evidence
    wheel = next(directory.glob('*.whl'))
    if count == 0:
        wheel.unlink()
    else:
        (directory / wheel.name.replace('0.1.0', '0.2.0')).write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError, match='exactly one'):
        verify_evidence(directory, run, **options)
