"""One command to check that everything the paper relies on is present, unchanged and reproducible.

    python scripts/check_submission.py                  # quick checks (~2 min)
    python scripts/check_submission.py --reproduce      # also re-run every result and compare (hours: GPU + CPU)
    python scripts/check_submission.py --reproduce results/aqfm_do3_s0_results.json   # only these

Quick checks: every results file is registered in scripts/manifest.json with the command that
makes it; data and model files match their recorded checksums; the commits that produced them
exist in git; results/TABLES.md is up to date; all tests pass; git is committed and pushed.

--reproduce re-runs the registered commands into reproduce_out/ (the committed results are
never touched) and compares every number with the committed file:
  IDENTICAL  every number equal
  NOISE      differences below 1e-6 relative (floating-point noise; reported numbers unchanged)
  DIFFERENT  the largest differences are listed
Prints ALL OK at the end, or the list of problems (exit code 1).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scripts", "manifest.json")
SKIP_KEYS = {"checkpoint", "code", "train_code", "args", "files"}   # paths and provenance, not results
problems = []


def report(ok, what, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        problems.append(what + (f": {detail}" if detail else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)


def check_manifest(m):
    print("Manifest: every result registered, with its models and the commits that made them")
    on_disk = {os.path.relpath(p, ROOT) for pat in ("results/*.json", "baselines/results/*.json")
               for p in glob.glob(os.path.join(ROOT, pat))}
    listed = set(m["results"])
    report(not on_disk - listed, "every results file is in the manifest", ", ".join(sorted(on_disk - listed)))
    report(not listed - on_disk, "every manifest result exists", ", ".join(sorted(listed - on_disk)))
    for path, r in m["results"].items():
        unknown = [k for k in r["models"] if k not in m["models"]]
        report(not unknown, f"{path}: models registered", ", ".join(unknown))
        ckpts = [t for t in shlex.split(r["run"]) if t.endswith(".pt")]
        expect = [m["models"][k]["path"] for k in r["models"] if k in m["models"]]
        report(ckpts == expect, f"{path}: command uses exactly its registered models",
               "" if ckpts == expect else f"{ckpts} vs {expect}")
    commits = {x["code"] for x in list(m["models"].values()) + list(m["results"].values())}
    missing = [c for c in sorted(commits) if git("cat-file", "-e", c + "^{commit}").returncode != 0]
    report(not missing, f"the {len(commits)} commits that produced them exist in git", ", ".join(missing))


def check_files(m):
    print("Data and models: present and unchanged (sha256)")
    for path, want in m["data"]["sha256"].items():
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            report(False, path, "missing; copy it from " + m["data"]["where"])
        else:
            report(sha256(full) == want, path, "" if sha256(full) == want else "checksum differs")
    for name, mod in m["models"].items():
        full = os.path.join(ROOT, mod["path"])
        where = f"Drive models/{os.path.basename(mod['drive'])}" if mod.get("drive") else "not on Drive"
        if not os.path.exists(full):
            report(False, f"model {name}", f"missing at {mod['path']} ({where})")
        else:
            ok = sha256(full) == mod["sha256"]
            report(ok, f"model {name} ({mod['path']})", "" if ok else "checksum differs")


def check_tables():
    print("Tables: results/TABLES.md matches the results files")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_tables
    path = os.path.join(ROOT, make_tables.OUT)
    fresh = make_tables.build()
    same = os.path.exists(path) and open(path).read() == fresh
    report(same, "results/TABLES.md is up to date", "" if same else "run: python scripts/make_tables.py")


def check_tests():
    print("Tests")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["no output"])[-1]
    report(r.returncode == 0, "all tests pass", last)


def check_git():
    print("Git: everything committed and pushed")
    dirty = git("status", "--porcelain").stdout.splitlines()
    report(not dirty, "no uncommitted or untracked files", f"{len(dirty)}: " + ", ".join(l[3:] for l in dirty[:8]))
    head = git("rev-parse", "HEAD").stdout.strip()
    remote = git("ls-remote", "origin", "refs/heads/main")
    if remote.returncode != 0 or not remote.stdout:
        print("  [WARN] could not reach GitHub to compare; check `git status` after `git fetch`")
    else:
        pushed = remote.stdout.split()[0] == head
        report(pushed, "GitHub main is at this commit", "" if pushed else "run: git push origin main")


def numbers(d, prefix=""):
    """Every number in a results file, keyed by its path (lists element-wise), provenance keys left out."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k not in SKIP_KEYS:
                yield from numbers(v, f"{prefix}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from numbers(v, f"{prefix}[{i}]")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix, float(d)


def compare(old_path, new_path):
    a, b = dict(numbers(json.load(open(old_path)))), dict(numbers(json.load(open(new_path))))
    missing = sorted(set(a) ^ set(b))
    diffs = sorted(((abs(a[k] - b[k]), k) for k in set(a) & set(b)
                    if a[k] != b[k] and not (math.isnan(a[k]) and math.isnan(b[k]))), reverse=True)
    if not missing and not diffs:
        return "IDENTICAL", f"{len(a)} numbers"
    noise = all(d <= 1e-6 * max(1.0, abs(a[k])) for d, k in diffs)
    if not missing and noise:
        return "NOISE", f"{len(diffs)} of {len(a)} numbers differ, largest {diffs[0][0]:.1e}"
    detail = [f"{k}: {a[k]:.4f} -> {b[k]:.4f}" for _, k in diffs[:5]]
    if missing:
        detail.append(f"{len(missing)} numbers only in one file, e.g. {missing[0]}")
    return "DIFFERENT", "; ".join(detail)


def reproduce(m, only, out_root):
    print(f"Reproduce: re-running {len(only)} result(s) into {os.path.relpath(out_root, ROOT)}/")
    os.makedirs(os.path.join(out_root, "logs"), exist_ok=True)
    if not os.path.exists(os.path.join(out_root, "data")):        # baselines read ../data from their cwd
        os.symlink(os.path.join(ROOT, "data"), os.path.join(out_root, "data"))
    done = set()
    for path in only:
        r = m["results"][path]
        cwd = os.path.join(ROOT, r.get("cwd", ""))
        tokens = []
        for t in shlex.split(r["run"]):
            if t == "python":
                t = sys.executable
            elif t == path or t in done:                              # its own output / an input re-made here
                t = os.path.join(out_root, t)
            elif t.endswith(".py") and os.path.exists(os.path.join(cwd, t)):
                t = os.path.join(cwd, t)
            tokens.append(t)
        run_cwd = os.path.join(out_root, r["cwd"]) if r.get("cwd") else ROOT
        os.makedirs(run_cwd, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(out_root, path)), exist_ok=True)
        log = os.path.join(out_root, "logs", path.replace("/", "__") + ".txt")
        print(f"  running {path} (~{r['minutes']} min) ... log: {os.path.relpath(log, ROOT)}", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.run(tokens, cwd=run_cwd, stdout=lf, stderr=subprocess.STDOUT).returncode
        new = os.path.join(out_root, path)
        if rc != 0 or not os.path.exists(new):
            report(False, f"reproduce {path}", f"command failed (exit {rc}); see the log")
            continue
        done.add(path)
        verdict, detail = compare(os.path.join(ROOT, path), new)
        report(verdict != "DIFFERENT", f"reproduce {path}: {verdict} ({(time.time() - t0) / 60:.0f} min)", detail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", nargs="*", metavar="RESULT", help="re-run these results (default: all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reproduce_out", time.strftime("%Y-%m-%d_%H%M")))
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    m = json.load(open(MANIFEST))
    check_manifest(m)
    check_files(m)
    check_tables()
    if not args.skip_tests:
        check_tests()
    check_git()
    if args.reproduce is not None:
        only = args.reproduce or list(m["results"])
        unknown = [p for p in only if p not in m["results"]]
        if unknown:
            sys.exit(f"not in the manifest: {unknown}")
        reproduce(m, [p for p in m["results"] if p in only], os.path.abspath(args.out))  # manifest order
    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
