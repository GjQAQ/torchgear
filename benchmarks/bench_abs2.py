"""Benchmark abs2 vs x.abs().square() on complex tensors."""

from __future__ import annotations

import sys

import torch
from torch.utils.benchmark import Compare, Timer

from torchgear import abs2  # noqa: E402

SHAPES = (
    (1 << 20,),
    (4096, 4096),
    (8, 4, 1024, 1024),
)
DTYPES = (torch.complex64, torch.complex128)

# GPU kernel launch dominates below this size; only assert at scale.
GPU_MIN_NUMEL = 1 << 24
# Allow small timing noise when both paths are bandwidth-bound on GPU.
GPU_MAX_SLOWDOWN = 1.02


def _make_complex(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    real = torch.randn(shape, device=device, dtype=torch.float32)
    imag = torch.randn(shape, device=device, dtype=torch.float32)
    return torch.complex(real, imag).to(dtype)


def _bench(x: torch.Tensor, *, label: str) -> list[Timer]:
    numel = x.numel()
    return [
        Timer(
            stmt='abs2(x)',
            globals={'abs2': abs2, 'x': x},
            label=label,
            sub_label='abs2',
            description=f'numel={numel}',
            num_threads=torch.get_num_threads(),
        ),
        Timer(
            stmt='x.abs().square()',
            globals={'x': x},
            label=label,
            sub_label='abs().square()',
            description=f'numel={numel}',
            num_threads=torch.get_num_threads(),
        ),
    ]


def _group_medians(measurements) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for m in measurements:
        key = (m.task_spec.label, m.task_spec.description)
        grouped.setdefault(key, {})[m.task_spec.sub_label] = m.median
    return grouped


def _check_performance(grouped: dict[tuple[str, str], dict[str, float]]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    skipped: list[str] = []

    for (label, desc), medians in sorted(grouped.items()):
        abs2_median = medians['abs2']
        baseline_median = medians['abs().square()']
        numel = int(desc.removeprefix('numel='))
        tag = f'{label} {desc}'

        if label.startswith('cpu'):
            if abs2_median >= baseline_median:
                failures.append(
                    f'{tag}: abs2 {abs2_median:.2e}s must be faster than '
                    f'abs().square() {baseline_median:.2e}s on CPU'
                )
            continue

        if numel < GPU_MIN_NUMEL:
            skipped.append(
                f'{tag}: GPU check skipped (numel < {GPU_MIN_NUMEL}); '
                f'abs2 {abs2_median:.2e}s vs abs().square() {baseline_median:.2e}s'
            )
            continue

        if abs2_median > baseline_median * GPU_MAX_SLOWDOWN:
            failures.append(
                f'{tag}: abs2 {abs2_median:.2e}s is slower than '
                f'abs().square() {baseline_median:.2e}s by more than '
                f'{(GPU_MAX_SLOWDOWN - 1) * 100:.0f}% on GPU'
            )

    return failures, skipped


def main() -> None:
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    timers: list[Timer] = []
    for device in devices:
        for dtype in DTYPES:
            for shape in SHAPES:
                x = _make_complex(shape, dtype, device)
                torch.testing.assert_close(abs2(x), x.abs().square())

                label = f'{device.type} {dtype}'
                timers.extend(_bench(x, label=label))

    measurements = [t.blocked_autorange() for t in timers]
    comparison = Compare(measurements)
    comparison.print()

    failures, skipped = _check_performance(_group_medians(measurements))

    if skipped:
        print('\nSkipped GPU checks (small tensors):')
        for msg in skipped:
            print(f'  - {msg}')

    if failures:
        print('\nFAILED:', file=sys.stderr)
        for msg in failures:
            print(f'  - {msg}', file=sys.stderr)
        sys.exit(1)

    print(
        '\nOK: CPU - abs2 faster than abs().square() for all shapes; '
        f'GPU - abs2 within {(GPU_MAX_SLOWDOWN - 1) * 100:.0f}% at numel >= {GPU_MIN_NUMEL}.'
    )


if __name__ == '__main__':
    main()
