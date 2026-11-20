"""Walk certificate intervals: run certdump.py FAMILY p0, read its valid interval [lo,hi], continue from just past the
end (direction +1 or -1) until the target end is reached or a step fails. Writes cover_<family>_<dir>.txt.
usage: cover.py FAMILY start end"""
import subprocess, sys, re, time
fam, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = 1 if end > start else -1
py = sys.executable
out = open('cover_%s_%s.txt' % (fam, 'up' if d > 0 else 'down'), 'a', encoding='utf-8')
p0 = start; covered = []
for it in range(60):
    t = time.time()
    r = subprocess.run([py, 'certdump.py', fam, '%.10f' % p0, '--quiet'], capture_output=True, text=True)
    m = re.search(r'valid on p in \[([\d.]+), ([\d.]+)\]\s+\(p0 cell valid: (\w+)\)', r.stdout)
    line = 'p0=%.10f rc=%d %.0fs ' % (p0, r.returncode, time.time() - t)
    if not m or m.group(3) != 'True':
        out.write(line + 'FAIL ' + (r.stdout[-300:] + r.stderr[-300:]).replace('\n', ' | ') + '\n'); out.flush()
        # nudge once inside the same cell
        p0 += d * 1e-4
        if abs(p0 - start) > abs(end - start): break
        continue
    lo, hi = float(m.group(1)), float(m.group(2))
    covered.append((lo, hi))
    out.write(line + 'valid [%.10f, %.10f]\n' % (lo, hi)); out.flush()
    nxt = hi if d > 0 else lo
    if (d > 0 and nxt >= end) or (d < 0 and nxt <= end): break
    p0 = nxt + d * 1e-6
out.write('DONE covered=%s\n' % covered); out.close()
