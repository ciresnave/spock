#!/usr/bin/env python3
"""Does a crate this workspace PUBLISHES sit on a version string the registry
already serves, while its content differs from what was served?

    version NOT on the registry  ->  no collision is possible          ->  OK
    version IS on the registry   ->  content MUST match the tarball    ->  DIVERGED

!! THE DEFECT IS A VERSION-STRING COLLISION, NOT DIVERGENCE.
A workspace member differing from the last published release is NORMAL -- it is
what development looks like. Reddening on that would be a detector firing on
healthy input, and a detector that fires on healthy input is worse than none.
The defect is sitting on a string the registry SERVES while differing from what
was served under it: a consumer resolving that string gets one artifact and this
workspace's own CI tests another, and both are green about different objects.

!! THE REMEDY IS A WORKFLOW, NOT A PUBLISH: bump immediately AFTER publishing,
not before. A member that moves to an unpublished version the moment it is
published can never collide, and the interval between publishes stops being a
period during which the tree makes a false claim.

!! THIS IS THE PRODUCER ARM AND IT IS THE ONLY ONE THAT FINDS THE DEFECT.
A lockfile tells you whether YOUR dependencies are pinned by checksum -- whether
you could be HURT by this class. It says nothing about whether YOU are one of
the two artifacts for somebody else. The candidate set here is every crate this
repo publishes, read from the workspace manifests; the lockfile is never opened.
A repo whose lockfile is entirely checksummed can still be a perpetrator.

Exit codes: 0 in report mode (the default) whatever it finds; with --gate, 1 if
anything DIVERGED. Always 1 if the scan itself found nothing to check -- a scan
that matches nothing must FAIL, not pass.

`--self-test` exercises the guards in this file against fabricated rows and
temporary directories; it needs no network and exits 1 if any arm fails.

!! Keep this docstring ASCII. argparse prints it for --help and a Windows
console is cp1252, so a warning glyph here makes --help crash with a
UnicodeEncodeError -- measured, on the first command anyone runs. The function
docstrings below are never printed and keep the usual markers.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

UA = {"User-Agent": "published-divergence-probe (+https://github.com/ciresnave/vulkane)"}
REGISTRY = "https://crates.io/api/v1/crates"


def members(manifest_dir: str) -> list[tuple[str, str, str]]:
    """Every PUBLISHABLE workspace member: (name, version, directory).

    From `cargo metadata`, never from Cargo.lock. `publish = false` members are
    excluded because they make no claim on a registry string.
    """
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=manifest_dir, capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        # stderr is kept deliberately: a swallowed failure here yields an empty
        # list, which reads as "nothing to check" rather than "cargo failed".
        sys.stderr.write(out.stderr)
        raise SystemExit("cargo metadata failed in %s" % manifest_dir)
    meta = json.loads(out.stdout)
    return [
        (p["name"], p["version"], os.path.dirname(p["manifest_path"]))
        for p in sorted(meta["packages"], key=lambda p: p["name"])
        if p.get("publish") != []
    ]


def served(name: str) -> set[str] | None:
    """Versions the registry serves. None means the crate was never published."""
    try:
        req = urllib.request.Request("%s/%s" % (REGISTRY, name), headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return {v["num"] for v in json.load(r)["versions"]}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch(name: str, version: str, into: str) -> str:
    url = "%s/%s/%s/download" % (REGISTRY, name, version)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        blob = r.read()
    path = os.path.join(into, "%s-%s.crate" % (name, version))
    io.open(path, "wb").write(blob)
    with tarfile.open(path, "r:gz") as t:
        # `filter="data"` is the 3.14 default and a hard error to omit there;
        # setting it explicitly keeps one behaviour across versions.
        try:
            t.extractall(into, filter="data")
        except TypeError:  # Python < 3.12 has no `filter`
            t.extractall(into)
    return os.path.join(into, "%s-%s" % (name, version))


def lines(path: str) -> list[str]:
    """CRLF-normalized. Without this every file differs on a Windows checkout,
    and a comparison that always fires is a comparison nobody reads."""
    raw = io.open(path, "rb").read().replace(b"\r\n", b"\n")
    return raw.decode("utf-8", errors="replace").split("\n")


def edit_size(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """(hunks, removed, added) from a real diff algorithm.

    ⚠️ NEVER a line-by-line inequality count. A positional walk marks every line
    after an insertion as differing: one file scored 689 that way and 15 by this
    one. First-divergence under-reports to a single token; positional comparison
    over-reports by everything downstream. Both are wrong about the change.
    """
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    return (len(ops),
            sum(o[2] - o[1] for o in ops),
            sum(o[4] - o[3] for o in ops))


def compare(published_dir: str, tree_dir: str) -> tuple[list[tuple], int]:
    """Authored source only.

    The tarball's Cargo.toml is cargo-normalized and its .cargo_vcs_info.json is
    generated, so both differ on every crate ever published. Including them would
    flag everything — the same always-fires failure as ignoring CRLF.
    """
    findings, checked = [], 0
    src = os.path.join(published_dir, "src")
    if not os.path.isdir(src):
        return findings, 0
    for base, _, files in os.walk(src):
        for f in files:
            pub = os.path.join(base, f)
            rel = os.path.relpath(pub, published_dir)
            tree = os.path.join(tree_dir, rel)
            checked += 1
            if not os.path.exists(tree):
                findings.append((rel, -1, 0, 0))
                continue
            a, b = lines(pub), lines(tree)
            if a != b:
                findings.append((rel,) + edit_size(a, b))
    return findings, checked


def report(rows: list[tuple], gate: bool) -> int:
    """Print the table and decide the exit code.

    Split out of `main` so the exit-code rules can be exercised with fabricated
    rows -- no network, no cargo, no registry. The anti-vacuous guard below is
    the one line that decides whether this file is a gate or a decoration, and
    a guard whose only proof lived in a throwaway directory is a guard nobody
    can re-check later.
    """
    print("  %-22s %-10s %-16s %6s  %s"
          % ("crate", "version", "string", "files", "verdict"))
    diverged = []
    for name, version, state, checked, findings in rows:
        verdict = "DIVERGED" if findings else "ok"
        if findings:
            diverged.append(name)
        print("  %-22s %-10s %-16s %6d  %s"
              % (name, version, state, checked, verdict))
        for rel, hunks, rem, add in findings:
            if hunks < 0:
                print("  %52s %s  (absent from the tree)"
                      % ("", rel.replace(os.sep, "/")))
            else:
                print("  %52s %s  %d hunks, -%d/+%d"
                      % ("", rel.replace(os.sep, "/"), hunks, rem, add))

    # A scan that matched nothing must FAIL, not pass. An empty member list and
    # a clean workspace produce identical silence otherwise, and the empty one
    # is the dangerous reading: it says "checked, all fine" about zero crates.
    if not rows:
        print("\n  !! no publishable members found -- the scan did not run.")
        return 1

    controls = [r for r in rows if r[2] in ("UNPUBLISHED", "NEVER-PUBLISHED")]
    print()
    if controls:
        print("  control: %s on an unpublished version -> ok, so a DIVERGED row is a"
              % controls[0][0])
        print("           finding rather than a comparator artifact.")
    else:
        print("  !! NO in-tree control: every member sits on a served version, so an")
        print("     all-ok result cannot be distinguished from a broken comparator.")
        print("     Bump one member post-publish to get a known-green row.")

    if diverged:
        print()
        print("  %d crate(s) sit on a SERVED version string with different content: %s"
              % (len(diverged), ", ".join(diverged)))
        print("  Remedy: bump to an unpublished version. Publishing is only needed if")
        print("  the difference is one consumers should receive -- measure that, do not")
        print("  assume it: identical sources are not required for identical behaviour,")
        print("  and differing sources do not imply differing output.")
        if gate:
            return 1
    return 0


def self_test() -> int:
    """Each arm below was run against a deliberately broken version of the code
    it checks, and failed there, before being kept.

    Every arm that expects a ZERO is downstream of the vacuous guard, so
    breaking that one guard reddens several arms at once; that is correct
    and not a redundancy to trim.

    Offline by construction except the last arm, which needs `cargo metadata`
    and says so. The point of keeping them here rather than in a scratch
    directory is the vacuous-scan path: a healthy workspace can never reach it,
    so nothing else will ever exercise it, and an unexercised guard is
    indistinguishable from an absent one.
    """
    failures = []

    def check(name, ok):
        print("  %-4s %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            failures.append(name)

    quiet = io.StringIO()

    def code(rows, gate):
        with contextlib.redirect_stdout(quiet):
            return report(rows, gate)

    CLEAN = [("a", "1.0.0", "SERVED", 12, []),
             ("b", "2.0.0", "UNPUBLISHED", 0, [])]
    DIRTY = CLEAN + [("c", "3.0.0", "SERVED", 4, [("src/lib.rs", 2, 5, 5)])]

    # -- the anti-vacuous guard, in BOTH modes ------------------------------
    # Report mode is where this lands first, so a vacuous scan has to fail
    # there too; otherwise the unarmed period is one in which the check cannot
    # report its own absence.
    check("an empty scan fails in report mode", code([], False) == 1)
    check("an empty scan fails in gate mode", code([], True) == 1)

    # -- and its control: a NON-empty clean scan must still pass -------------
    # Without this the guard could be an unconditional `return 1` and both
    # arms above would still read as passes.
    check("a clean non-empty scan passes when armed", code(CLEAN, True) == 0)

    # -- mode discrimination -------------------------------------------------
    check("a divergence reports without blocking when unarmed",
          code(DIRTY, False) == 0)
    check("the same divergence blocks when armed", code(DIRTY, True) == 1)

    with tempfile.TemporaryDirectory() as tmp:
        crlf = os.path.join(tmp, "crlf.rs")
        lf = os.path.join(tmp, "lf.rs")
        other = os.path.join(tmp, "other.rs")
        io.open(crlf, "wb").write(b"fn a() {}\r\nfn b() {}\r\n")
        io.open(lf, "wb").write(b"fn a() {}\nfn b() {}\n")
        io.open(other, "wb").write(b"fn a() {}\nfn c() {}\n")

        # A comparison that fires on every file is a comparison nobody reads,
        # and on a Windows checkout line endings alone produce exactly that.
        check("line endings alone are not a difference", lines(crlf) == lines(lf))
        # ...and its control, or the normalizer could be returning a constant.
        check("a real difference survives normalization", lines(lf) != lines(other))

        # `edit_size` must not be a positional walk. Every line after the
        # insertion shifts, so a zip-and-count says 3; the answer is 1 hunk of
        # 1 added line. Both numbers are computed here, so the docstring's
        # claim about the two instruments is checked rather than asserted.
        a = ["one", "two", "three", "four"]
        b = ["one", "INSERTED", "two", "three", "four"]
        hunks, rem, add = edit_size(a, b)
        positional = sum(1 for x, y in zip(a, b) if x != y)
        check("an insertion is one hunk, not everything downstream",
              (hunks, rem, add) == (1, 0, 1) and positional == 3)

        pub, tree = os.path.join(tmp, "pub"), os.path.join(tmp, "tree")
        os.makedirs(os.path.join(pub, "src"))
        os.makedirs(os.path.join(tree, "src"))
        io.open(os.path.join(pub, "src", "lib.rs"), "wb").write(b"same\n")
        io.open(os.path.join(tree, "src", "lib.rs"), "wb").write(b"same\n")
        found, checked = compare(pub, tree)
        check("identical sources yield no findings, over a nonzero file count",
              found == [] and checked == 1)

        io.open(os.path.join(pub, "src", "gone.rs"), "wb").write(b"x\n")
        found, checked = compare(pub, tree)
        check("a file the tarball has and the tree lacks is flagged",
              len(found) == 1 and found[0][1] == -1 and checked == 2)

        # -- the one arm that needs a toolchain ----------------------------
        # `cargo metadata --no-deps` does not resolve dependencies, so this
        # stays offline. It answers first: a result from a cargo that never
        # identified itself says nothing about which cargo produced it.
        # A missing cargo must say so in its own words. Left bare it raises
        # FileNotFoundError and the arm reds with a traceback, which reads as a
        # defect in this file rather than as a runner without a toolchain --
        # and this job installs none, relying on the image providing one.
        try:
            ver = subprocess.run(["cargo", "--version"], capture_output=True,
                                 text=True, encoding="utf-8")
            answer = (ver.stdout or ver.stderr).strip()
            rc = ver.returncode
        except OSError as e:
            answer, rc = "NOTHING (%s)" % e.__class__.__name__, 1
        print("  --   cargo answers: %s" % (answer or "NOTHING"))
        if rc != 0:
            print("       ^ this arm needs a toolchain on the runner; add one to")
            print("         the job rather than reading the failure as a code defect.")
        ws = os.path.join(tmp, "ws")
        os.makedirs(os.path.join(ws, "src"))
        io.open(os.path.join(ws, "Cargo.toml"), "w", encoding="utf-8").write(
            '[package]\nname = "unpublishable"\nversion = "0.1.0"\n'
            'edition = "2021"\npublish = false\n\n[workspace]\n')
        io.open(os.path.join(ws, "src", "main.rs"), "w",
                encoding="utf-8").write("fn main() {}\n")
        check("a `publish = false` member is not a candidate",
              rc == 0 and members(ws) == [])

    if failures:
        print("\n  self-test FAILED: %s" % "; ".join(failures))
        return 1
    print("\n  self-test passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", default=".", help="workspace root")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 on divergence (default: report only)")
    ap.add_argument("--self-test", action="store_true",
                    help="check this file's own guards; needs no network")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, version, tree_dir in members(args.manifest_dir):
            pub = served(name)
            if pub is None:
                rows.append((name, version, "NEVER-PUBLISHED", 0, []))
            elif version not in pub:
                rows.append((name, version, "UNPUBLISHED", 0, []))
            else:
                d = fetch(name, version, tmp)
                findings, checked = compare(d, tree_dir)
                rows.append((name, version, "SERVED", checked, findings))

    return report(rows, args.gate)


if __name__ == "__main__":
    sys.exit(main())
