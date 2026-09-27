"""Archive emitted gfx906 ISA and resource metadata for the measured kernels."""
import argparse
import gzip
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / 'tools'))
import kernel_resources as resources

parser = argparse.ArgumentParser()
parser.add_argument('binary', type=pathlib.Path)
parser.add_argument('prefix')
args = parser.parse_args()
output = pathlib.Path(__file__).resolve().parent
records = []
for co in resources.code_objects(str(args.binary), 'gfx906'):
    kernels = resources.kernels(co)
    names = resources.demangle([k['name'] for k in kernels])
    for k, name in zip(kernels, names):
        if 'activation_prepare::step_kernel<' in name:
            mode = re.search(r'Output\)(\d)', name)[1]
            label = 'step-' + mode
        elif 'activation_prepare::q4_half_gemv_kernel<4096u' in name:
            label = 'half-gemv'
        else:
            continue
        with tempfile.NamedTemporaryFile(suffix='.co') as f:
            f.write(co)
            f.flush()
            isa = subprocess.run(['/opt/rocm/lib/llvm/bin/llvm-objdump', '-d', '--mcpu=gfx906', '--disassemble-symbols=' + k['name'], f.name], check=True, capture_output=True, text=True).stdout
            isa = isa.replace(f.name, '<code-object>')
        archive = output / f'{args.prefix}-{label}-isa.txt.gz'
        with archive.open('wb') as f:
            with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as z:
                z.write(isa.encode())
        instructions = [line.strip() for line in isa.splitlines() if any(op in line for op in ('v_cvt_f16_f32', 'flat_store_dword', 'flat_load_dword', 'global_store_short'))]
        (output / f'{args.prefix}-{label}-selected-isa.txt').write_text('\n'.join([name, *instructions]) + '\n')
        records.append({'kernel': name, 'resources': k, 'code_object_sha256': hashlib.sha256(co).hexdigest(), 'isa_archive': archive.name, 'cvt_f16_f32_count': len(re.findall(r'\bv_cvt_f16_f32(?:_|\s)', isa)), 'isa_sha256': hashlib.sha256(isa.encode()).hexdigest()})
(output / (args.prefix + '-isa-manifest.json')).write_text(json.dumps({'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'kernels': records}, indent=2) + '\n')
print(f'{args.prefix}: {len(records)} kernels archived')
