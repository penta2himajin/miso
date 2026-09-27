"""Sequential matched native/timestamp-traced process pairs with ADR_002 telemetry."""
import gzip
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
HARNESS = REPO / 'bench/model/results/2026-09-26-decode-investigation/run_measure.py'
BINARY = REPO / 'build/bench/bench_norm_chain'
WORK = pathlib.Path('/tmp/miso-norm-latency')
ARGS = ['--iterations', '128', '--rounds', '3', '--warmup', '128']


def main():
    WORK.mkdir(exist_ok=True)
    for pair in range(1, 4):
        for traced in ([False, True] if pair % 2 else [True, False]):
            label = 'traced' if traced else 'native'
            log = HERE / f'matched-{pair}-{label}.txt'
            prefix = WORK / f'matched-{pair}-traced'
            cmd = [str(BINARY), *ARGS]
            if traced:
                cmd = ['/opt/rocm/bin/rocprof', '--timestamp', 'on', '--stats',
                       '-o', str(prefix) + '.csv', *cmd]
            subprocess.run([sys.executable, str(HARNESS), str(log), '1', *cmd],
                           cwd=REPO, check=True)
            if traced:
                for suffix in ['csv', 'stats.csv', 'sysinfo.txt']:
                    src = pathlib.Path(str(prefix) + '.' + suffix)
                    with gzip.open(HERE / (prefix.name + '.' + suffix + '.gz'), 'wb') as out:
                        out.write(src.read_bytes())


if __name__ == '__main__':
    main()
