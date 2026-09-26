import datetime, json, pathlib, subprocess, sys, threading, time
out = pathlib.Path(sys.argv[1]); count = int(sys.argv[2]); cmd = sys.argv[3:]
out.parent.mkdir(parents=True, exist_ok=True)
stop = threading.Event()
def smi():
    with open(str(out)+'.smi.jsonl', 'w') as f:
        while not stop.is_set():
            r = subprocess.run(['/opt/rocm/bin/rocm-smi','--showclocks','--showtemp','--showpower','--showuse','--json'], capture_output=True, text=True)
            f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data':r.stdout,'stderr':r.stderr})+'\n'); f.flush()
            stop.wait(0.5)
t = threading.Thread(target=smi); t.start()
try:
    with open(out, 'w') as f:
        f.write('command: '+repr(cmd)+'\n'); f.flush()
        for i in range(count):
            f.write(f'run {i+1} utc {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n'); f.flush()
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            f.write(f'exit: {r.returncode}\n'); f.flush()
            print(f'{out}: run {i+1} exit {r.returncode}', flush=True)
            if r.returncode: raise SystemExit(r.returncode)
finally:
    stop.set(); t.join()
