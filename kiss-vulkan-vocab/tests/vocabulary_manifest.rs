//! The committed §6.8-0008 manifest must be what this crate emits today, and
//! must satisfy the envelope KISS pins.
//!
//! # The freshness gate
//!
//! §6.8-0011 requires a manifest to "**agree** with its prose annex under an
//! emit-and-`git diff --exit-code` freshness gate". This file is the emit half,
//! run as a test so it fails in CI rather than only when someone remembers to
//! regenerate.
//!
//! It `#[path]`-includes the generator rather than re-implementing it or
//! shelling out to `cargo run`. Re-implementing would compare the artifact
//! against a second copy of the logic, which is what an emit-and-compare gate
//! exists to rule out; shelling out would make the test depend on a nested
//! cargo invocation holding the build lock.
//!
//! # What this file does NOT close
//!
//! **Agreement with `spec/namespaces/vulkan.md` is a separate obligation and is
//! not tested here.** §6.8-0011 splits provenance from agreement — *"Provenance
//! names the producer; agreement is a relation between two artifacts, and
//! neither settles which is the source."* This gate proves the manifest is
//! fresh against the crate. It does not prove the crate agrees with the annex,
//! which is the gap `registered_namespace.rs` calls "a ratchet, not a proof".
//! Saying so here rather than letting a green run imply otherwise.

#[path = "../examples/emit_vocabulary_manifest.rs"]
// The example is compiled INTO this test so the emitter is exercised rather
// than a copy of it. The test calls a subset of what the example defines, and
// the rest is live in the example binary -- so this says "unused by this
// test", not "unused".
#[allow(dead_code)]
mod emitter;

use std::path::PathBuf;

fn committed_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("manifest")
        .join("vulkan-vocabulary.json")
}

/// The committed manifest, **exactly as it sits on disk**.
///
/// This used to normalize `\r\n` before comparing, and that normalization was
/// hiding a real defect rather than tolerating a cosmetic one. `.gitattributes`
/// did not pin this path, so a Windows checkout stored LF and checked out CRLF
/// — the committed artifact was **not** byte-identical to a fresh emission, and
/// the emit-and-`git diff --exit-code` gate KISS-CLASSIFY-6.8-0011 asks for
/// could not have been armed here at all. The test passed the whole time,
/// because it was comparing something neither party actually had.
///
/// The path is pinned `text eol=lf` now, so the comparison can be exact.
fn committed() -> String {
    let p = committed_path();
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read the committed manifest at {}: {e}", p.display()))
}

/// The emit-and-compare freshness gate.
#[test]
fn committed_manifest_is_byte_identical_to_a_fresh_emission() {
    let fresh = emitter::manifest();
    let on_disk = committed();

    // Line endings get their own message. Reporting "stale" for a CRLF checkout
    // would send the reader hunting for a content change that does not exist,
    // and the fix is completely different from the fix for real staleness.
    if fresh != on_disk && fresh == on_disk.replace("\r\n", "\n") {
        // The claim is exactly what the condition above tested: the two agree
        // once CRLF is folded to LF. That admits a mixed-ending file as well as
        // a uniformly-CRLF one, so the message says "contains CRLF" rather than
        // "is CRLF" — the wider claim would be true in the common case and
        // wrong in the one that is harder to diagnose.
        let crlf = on_disk.matches("\r\n").count();
        panic!(
            "the committed manifest differs from a fresh emission ONLY in line \
             endings — it contains {crlf} CRLF line ending(s) and the emitter \
             produces LF.\n\n\
             The content is fine. `.gitattributes` should be pinning\n  \
             kiss-vulkan-vocab/manifest/*.json text eol=lf\n\
             and this checkout is not honouring it. Re-materialize the file:\n  \
             rm {} && git checkout -- {}\n\n\
             This is not cosmetic: the artifact is byte-compared, and a copy \
             carrying CRLF cannot satisfy an emit-and-`git diff --exit-code` \
             gate.",
            committed_path().display(),
            committed_path().display()
        );
    }

    if fresh != on_disk {
        // Report the first divergence rather than dumping 23KB of JSON at
        // someone — a diff nobody reads is a failure message that only says
        // "something changed".
        // Byte offset of the first difference, then windows clamped OUTWARD to
        // char boundaries. This manifest is full of em-dashes, so a raw
        // `at ± 60` lands mid-character often, and the old version fell back to
        // the literal string "<boundary>" when it did — a diagnostic that
        // silently degrades exactly when it is needed.
        let at = fresh
            .as_bytes()
            .iter()
            .zip(on_disk.as_bytes())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| fresh.len().min(on_disk.len()));
        let window = |s: &str| {
            let mut start = at.saturating_sub(60).min(s.len());
            while start > 0 && !s.is_char_boundary(start) {
                start -= 1;
            }
            let mut end = (at + 60).min(s.len());
            while end < s.len() && !s.is_char_boundary(end) {
                end += 1;
            }
            s[start..end].replace('\n', "⏎")
        };
        panic!(
            "the committed vocabulary manifest is stale.\n\n\
             First divergence at byte {at}.\n  emitted:   …{}…\n  committed: …{}…\n\n\
             Regenerate it in the same change that altered the vocabulary:\n  \
             cargo run --example emit_vocabulary_manifest -p kiss-vulkan-vocab \\\n    \
             > kiss-vulkan-vocab/manifest/vulkan-vocabulary.json\n\n\
             Do not edit the manifest by hand. It is the machine-readable form \
             of a vocabulary another project binds against, and a hand-edit \
             makes it disagree with the crate that is supposed to produce it — \
             which is the drift this gate exists to catch.",
            window(&fresh),
            window(&on_disk)
        );
    }
}

/// §6.8-0008's envelope: every required field present, `schema` recognised.
#[test]
fn manifest_carries_every_field_the_envelope_requires() {
    let m = committed();
    for key in [
        "schema",
        "namespace",
        "vocabulary_version",
        "generated_from",
        "kind",
        "grammar",
        "coverage_note",
    ] {
        assert!(
            m.contains(&format!("\"{key}\":")),
            "the manifest is missing the required envelope field {key:?}. \
             §6.8-0008 lists these explicitly and says a reader MUST reject \
             with a typed decline a manifest missing any of them — so omitting \
             one does not degrade the artifact, it invalidates it."
        );
    }
    assert!(
        m.contains("\"schema\": \"kiss-namespace-vocabulary-v1\""),
        "the manifest's schema id is not `kiss-namespace-vocabulary-v1`; a \
         reader MUST reject an unrecognized schema."
    );
    assert!(
        m.contains("\"kind\": \"generated\""),
        "`vulkan` is a grammar over an open product space, so its kind is \
         `generated`. §6.8-0010 makes `kind` an OPEN set and requires a reader \
         encountering an unknown one to decline rather than guess the nearer \
         of the two it knows."
    );
}

/// `vocabulary_version` must be an **integer**, and the check must be able to
/// fail on a float.
///
/// §6.8-0008 states the reason inline: *"an integer — a gate that truncates a
/// fractional value is not a gate."* A clause that anticipates its own defeat
/// deserves a test that does too, so this asserts the emitted form is an
/// integer literal **and** demonstrates on a fabricated float that the check
/// rejects it. Asserting only the happy path would leave a check that cannot
/// tell the two apart.
#[test]
fn vocabulary_version_is_an_integer_and_a_float_would_be_rejected() {
    let m = committed();

    let value = m
        .split("\"vocabulary_version\":")
        .nth(1)
        .and_then(|t| t.split(',').next())
        .map(str::trim)
        .expect("manifest carries a vocabulary_version");

    assert!(
        is_integer_literal(value),
        "vocabulary_version is {value:?}, which is not an integer literal. \
         §6.8-0008: \"a gate that truncates a fractional value is not a gate.\" \
         A quoted value fails for the same reason — a consumer comparing it \
         numerically would parse it first, and a parse that truncates is the \
         defeat the clause names."
    );
    assert_eq!(
        value,
        kiss_vulkan_vocab::VOCABULARY_VERSION.to_string(),
        "the manifest's vocabulary_version disagrees with the crate's"
    );

    // Negative controls: the predicate must reject what the clause warns about.
    for bad in ["4.0", "4.5", "\"4\"", "4e0", " 4 .0", "+4", "0x4"] {
        assert!(
            !is_integer_literal(bad),
            "is_integer_literal accepted {bad:?}; the gate would truncate it \
             and report success, which is exactly the failure §6.8-0008 names"
        );
    }
    for good in ["0", "4", "17", "4294967295"] {
        assert!(
            is_integer_literal(good),
            "is_integer_literal rejected {good:?}, which is a valid version"
        );
    }
}

/// A bare decimal integer: no sign, no point, no exponent, no quotes, no radix
/// prefix, and no leading zero on a multi-digit value.
fn is_integer_literal(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) && (s.len() == 1 || !s.starts_with('0'))
}

/// §6.8-0013: for `kind: generated` the vectors are the normative contract, and
/// the required coverage is enumerated. A namespace with no length-conditional
/// field may omit `threshold`/`digest_input` **and must say so** — `vulkan` has
/// two, so both must be present for both.
#[test]
fn vectors_cover_every_canonicalization_the_clause_requires() {
    let m = committed();

    for pins in ["order", "dedup", "threshold", "digest_input"] {
        assert!(
            m.contains(&format!("\"pins\": \"{pins}\"")),
            "no vector pins {pins:?}. §6.8-0013 enumerates the required \
             coverage for `kind: generated`, and a missing tag is a coverage \
             hole rather than a smaller vector set: the grammar cannot validate \
             canonicalization, so whatever the vectors omit is unpinned."
        );
    }

    // Both length-conditional fields, both sides of the boundary, both digests.
    for field in ["coop", "coopvec"] {
        // ⚠️ Counts BOTH spellings. A threshold vector names its field with
        // `threshold_of` (KISS-CLASSIFY-6.8-0016, merged 2026-09-06); every other
        // vector kind still uses `field`, because `threshold_of` on a vector that
        // pins no boundary would be a category error.
        //
        // This test broke the moment that rename landed, which is the point worth
        // recording: it is a READER KEYED ON THE OLD NAME, in the same repository
        // as the emitter, and nothing connected the two but a string. That is the
        // failure mode the clause rename exists to prevent between projects,
        // reproduced inside one crate within minutes.
        let count = m.match_indices(&format!("\"field\": \"{field}\"")).count()
            + m.match_indices(&format!("\"threshold_of\": \"{field}\""))
                .count();
        assert!(
            count >= 5,
            "field {field:?} has only {count} vectors; expected at least five \
             (order, dedup, threshold-at, threshold-across, digest_input). The \
             two length-conditional fields measure and digest INDEPENDENTLY, so \
             covering one does not cover the other — an implementation that \
             switched `coop` correctly and `coopvec` early would pass a \
             single-field vector set."
        );
    }

    assert!(
        m.contains("\"enumeration_bytes\": 512") && m.contains("\"enumeration_bytes\": 513"),
        "the threshold vectors do not sit at 512 and 513 bytes. §6.8-0013 wants \
         each length-conditional field presented AT and IMMEDIATELY ACROSS its \
         boundary, \"so both forms are pinned at the exact byte count that flips \
         them\". A straddling pair that never lands on the boundary cannot tell \
         `>` from `>=`."
    );
}

/// The digest is over the pinned `digest_input`, and the pinned input is the
/// same string the threshold measured.
///
/// §6.8-0013 wants this separable from the threshold "so a producer may
/// disagree about *whether* to digest but never about *what* is digested".
/// Those are different failures and only one of them is visible in the token —
/// the token carries the hash, so a producer digesting the wrong string emits a
/// well-formed token that matches nothing.
#[test]
fn each_digest_is_the_hash_of_the_digest_input_it_pins() {
    let m = committed();
    let mut checked = 0;

    for chunk in m.split("\"pins\": \"digest_input\"").skip(1) {
        let entry = chunk.split('}').next().unwrap_or_default();
        let field = between(entry, "\"digest_input\": \"", "\", \"digest_input_bytes\"")
            .expect("digest_input vector carries its input string");
        let declared_len: usize = between(entry, "\"digest_input_bytes\": ", ",")
            .and_then(|s| s.trim().parse().ok())
            .expect("digest_input vector carries its byte count");
        let digest =
            between(entry, "\"digest\": \"", "\"").expect("digest_input vector carries a digest");

        let unescaped = field.replace("\\\"", "\"").replace("\\\\", "\\");
        assert_eq!(
            unescaped.len(),
            declared_len,
            "a digest_input vector declares {declared_len} bytes but carries \
             {}; the length a consumer measures against the threshold and the \
             string it hashes must be the same string",
            unescaped.len()
        );
        assert_eq!(
            digest,
            format!(
                "fnv1a64-{:016x}",
                kiss_vulkan_vocab::fnv1a64(unescaped.as_bytes())
            ),
            "a digest_input vector's digest is not the FNV-1a-64 of the input \
             it pins. This is the one disagreement invisible in a token — the \
             token carries only the hash, so a producer that digests the wrong \
             string emits a well-formed token matching nothing."
        );
        checked += 1;
    }

    assert_eq!(
        checked, 2,
        "expected one digest_input vector per length-conditional field, found \
         {checked}. `vulkan` has two such fields and they digest independently."
    );
}

fn between<'a>(hay: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = hay.find(start)? + start.len();
    let rest = &hay[s..];
    let e = rest.find(end)?;
    Some(&rest[..e])
}

/// The committed manifest must satisfy KISS-CLASSIFY-6.8-0016's own rejection
/// conditions, checked here rather than trusted.
///
/// §6.8-0016 (merged 2026-09-06) requires that a `threshold`-tagged vector carry
/// `threshold_of` and `enumeration_bytes`, and that **a reader MUST reject** a
/// manifest in which, for any value of `threshold_of`, the threshold vectors do
/// not include a pair whose `enumeration_bytes` are N and N+1, **or** in which
/// that pair's two emitted `token` values are equal.
///
/// ⚠️ This asserts the conditions a KISS reader will apply to us, from our side,
/// so a divergence fails here rather than in somebody else's decline. The clause
/// exists because **adjacency does not establish straddling** — enumerations of
/// 3 and 4 bytes are adjacent and both far below a 512-byte boundary — and the
/// differing-token condition is what makes the byte pair mean anything: a
/// declared boundary that flips no behaviour is a wrong boundary.
///
/// Hand-parsed: this crate has no dependencies, dev-dependencies included, which
/// §6.9-0003 requires and `zero_dependency.rs` enforces.
#[test]
fn threshold_vectors_straddle_their_boundary_per_6_8_0016() {
    let text = std::fs::read_to_string(committed_path()).expect("committed manifest");

    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        let at = line.find(&format!("\"{key}\": "))? + key.len() + 4;
        let rest = &line[at..];
        Some(if let Some(r) = rest.strip_prefix('"') {
            &r[..r.find('"')?]
        } else {
            let end = rest.find([',', '}']).unwrap_or(rest.len());
            rest[..end].trim()
        })
    }

    let rows: Vec<(&str, u64, &str)> = text
        .lines()
        .filter(|l| l.contains("\"pins\": \"threshold\""))
        .map(|l| {
            let of = field(l, "threshold_of").unwrap_or_else(|| {
                panic!(
                    "a threshold vector without `threshold_of`; §6.8-0016 makes it MUST: {l:.120}"
                )
            });
            let bytes: u64 = field(l, "enumeration_bytes")
                .unwrap_or_else(|| panic!("threshold vector without `enumeration_bytes`: {l:.120}"))
                .parse()
                .expect("enumeration_bytes is a number");
            let token = field(l, "token")
                .unwrap_or_else(|| panic!("threshold vector without `token`: {l:.120}"));
            (of, bytes, token)
        })
        .collect();

    // Positive control: a parser that silently matched nothing would satisfy
    // every assertion below by having nothing to check.
    assert!(
        rows.len() >= 2,
        "found {} threshold vectors; the manifest has length-conditional fields, so \
         too few means the parser broke rather than the manifest shrank",
        rows.len()
    );

    let mut fields: Vec<&str> = rows.iter().map(|(f, _, _)| *f).collect();
    fields.sort_unstable();
    fields.dedup();
    for f in fields {
        let mut group: Vec<&(&str, u64, &str)> = rows.iter().filter(|(o, _, _)| *o == f).collect();
        group.sort_by_key(|(_, b, _)| *b);
        let pair = group
            .windows(2)
            .find(|w| w[1].1 == w[0].1 + 1)
            .unwrap_or_else(|| {
                panic!(
                    "threshold_of={f:?} has no N/N+1 pair; byte counts are {:?}. \
                     Adjacency in the LIST is not adjacency in the BYTES -- a reader \
                     MUST reject this under §6.8-0016.",
                    group.iter().map(|(_, b, _)| *b).collect::<Vec<_>>()
                )
            });
        assert_ne!(
            pair[0].2, pair[1].2,
            "threshold_of={f:?}: the N/N+1 pair at {} and {} emits the SAME token, so the \
             declared boundary flips nothing and is a wrong boundary",
            pair[0].1, pair[1].1
        );
    }
}

/// The manifest must satisfy KISS-CLASSIFY-6.8-0017's rejection conditions,
/// checked here rather than trusted.
///
/// A reader MUST reject a manifest whose `sufficiency` is absent, whose
/// `status` is **absent** or is any other token, or which claims `demonstrated`
/// without all five of `reproduced_by`, `artifact`, `vocabulary_version`,
/// `guessed` and `derived`.
///
/// ⚠️ The absent-`status` arm is checked SEPARATELY from the wrong-token arm,
/// mirroring the clause's own reason for stating them separately: an
/// enumeration of wrong values does not reach a value that is not there. That
/// distinction is not pedantry — it is how §6.8-0017 came to mandate a field it
/// never named, and the clause says so about itself.
///
/// Hand-parsed: this crate has no dependencies, dev-dependencies included.
#[test]
fn sufficiency_is_declared_per_6_8_0017() {
    let text = std::fs::read_to_string(committed_path()).expect("committed manifest");

    let start = text
        .find("\"sufficiency\"")
        .expect("§6.8-0017: `sufficiency` is absent. A reader MUST reject this.");
    let block: String = text[start..]
        .lines()
        .take_while(|l| !l.trim_start().starts_with("},"))
        .collect::<Vec<_>>()
        .join("\n");
    // Positive control: a slice that captured nothing would satisfy an
    // absence-based check by having nothing to contradict it.
    assert!(
        block.len() > 30 && block.contains('{'),
        "the sufficiency block did not parse out ({block:?}); the extractor broke \
         rather than the manifest shrinking"
    );

    let value = |key: &str| -> Option<String> {
        let at = block.find(&format!("\"{key}\": "))? + key.len() + 4;
        let rest = &block[at..];
        Some(if let Some(r) = rest.strip_prefix('"') {
            r[..r.find('"')?].to_owned()
        } else {
            rest[..rest.find([',', '\n']).unwrap_or(rest.len())]
                .trim()
                .to_owned()
        })
    };

    // Arm 1: absent. Stated apart from arm 2 on purpose.
    let status = value("status").expect(
        "§6.8-0017: `sufficiency` carries no `status`. This is the ABSENT arm, and it is \
         the one an enumeration of wrong tokens does not reach.",
    );
    // Arm 2: any other token.
    assert!(
        status == "demonstrated" || status == "unexercised",
        "§6.8-0017: `status` is {status:?}; exactly `demonstrated` or `unexercised`"
    );

    // Arm 3: `demonstrated` without all five.
    if status == "demonstrated" {
        for k in [
            "reproduced_by",
            "artifact",
            "vocabulary_version",
            "guessed",
            "derived",
        ] {
            assert!(
                value(k).is_some() || block.contains(&format!("\"{k}\"")),
                "§6.8-0017: `status` is `demonstrated` without `{k}`. All five are \
                 required, and `guessed`/`derived` MAY be empty but MUST be present \
                 — an empty array is the strong claim, and a reader is entitled to \
                 see it made."
            );
        }
    }

    // Arm 4: the note must NAME the vector count this manifest actually carries.
    //
    // ⚠️ Not hypothetical. The first `sufficiency` note said "This manifest
    // has thirteen" and was wrong one commit later, when three vectors landed
    // and `vocabulary_version` did not move. The block whose entire purpose is
    // to say "what ships here is not what was reproduced" had gone stale about
    // what ships here.
    //
    // Stated as a POSITIVE requirement rather than a bound. The first draft of
    // this arm asserted `n <= vectors` over every digit run in the note, which
    // a stale `13` satisfies as comfortably as a correct `16` — a gate that
    // cannot fail the defect it is named after. Requiring the true count to
    // APPEAR has no such hole: there is exactly one number that satisfies it.
    let vectors = text.matches("\"pins\": ").count();
    assert!(
        vectors > 0,
        "positive control: the vector extractor found none, so the check below \
         would be comparing against zero and any note at all would pass it"
    );
    let named: Vec<usize> = block
        .split(|c: char| !c.is_ascii_digit())
        .filter(|w| !w.is_empty())
        .filter_map(|w| w.parse().ok())
        .collect();
    assert!(
        named.contains(&vectors),
        "the sufficiency note names {named:?} but this manifest carries \
         {vectors} vectors, and the note must say so. A number here is a claim \
         about WHICH ARTIFACT is shipping; a stale one asserts a reproduction \
         of something that is not being shipped — the exact failure §6.8-0017 \
         exists to prevent, arriving from inside the field meant to prevent it."
    );
}

/// The three vectors closing baracuda's residue pin SPELLINGS, and a note that
/// describes a spelling is prose until something compares it to the token.
///
/// ⚠️ Written because the first draft of the `saturating` note asserted the
/// OPPOSITE of what this vocabulary does. It claimed `saturating` was not
/// spelled into the tuple and that two shapes differing only in that field
/// would collapse into one; the emitted token spells a trailing `-sat` and
/// keeps both. The note was wrong for the same reason the gap existed — no
/// vector had ever produced the token, so nothing could contradict a
/// plausible sentence about it.
///
/// A vector is self-consistent BY CONSTRUCTION here: the emitter derives the
/// token from the same code that spells it, so input and token can never
/// disagree. That is deliberate, and it is also why a vector cannot catch a
/// wrong NOTE. This test is the only thing standing between the two halves.
#[test]
fn flag_vectors_pin_the_spellings_their_notes_describe() {
    let m = committed();

    let line_for = |pins: &str| -> String {
        m.lines()
            .find(|l| l.contains(&format!("\"pins\": \"{pins}\"")))
            .unwrap_or_else(|| {
                panic!(
                    "no vector pins {pins:?}. This is the ABSENT arm: the test \
                     cannot check a spelling that no vector produces, and a \
                     silently-skipped check is what let the wrong note ship."
                )
            })
            .to_string()
    };
    let token_for = |pins: &str| -> String {
        let line = line_for(pins);
        between(&line, "\"token\": \"", "\"")
            .unwrap_or_else(|| panic!("vector {pins:?} carries no token"))
            .to_string()
    };
    let coop_tuples = |token: &str| -> Vec<String> {
        token
            .split('.')
            .find(|p| p.starts_with("cm-"))
            .unwrap_or_else(|| panic!("token {token:?} has no <coop> field"))
            .trim_start_matches("cm-")
            .split(',')
            .map(str::to_string)
            .collect()
    };

    // -- subgroup: the dynamic width, in the TOKEN and in the INPUT.
    let sg = token_for("subgroup");
    assert!(
        sg.starts_with("vulkan:sgdyn."),
        "the subgroup vector's token is {sg:?}, which does not spell `sgdyn`. \
         The width-agnostic case is the one spelling `<subgroup>` has that is \
         not a number, so a token carrying a width here pins nothing new."
    );
    assert!(
        line_for("subgroup").contains("\"subgroup\": \"dynamic\""),
        "the subgroup vector's INPUT does not spell the dynamic case as the \
         string \"dynamic\". The gap this vector closes is in the input half — \
         a reader who has only seen `\"subgroup\": 32` cannot know what to pass \
         — so an input spelled any other way leaves the gap open."
    );

    // -- saturating: spelled as a trailing `-sat`, and NOT collapsed.
    let tuples = coop_tuples(&token_for("saturating"));
    assert_eq!(
        tuples.len(),
        2,
        "the saturating vector spells {} <coop> tuple(s), expected 2. Two \
         shapes differing only in `saturating` are DISTINCT, and a producer \
         that dropped the field would emit one tuple where this emits two — \
         under byte-exact matching, a different cell rather than a near miss.",
        tuples.len()
    );
    assert_eq!(
        tuples.iter().filter(|t| t.ends_with("-sat")).count(),
        1,
        "expected exactly one of {tuples:?} to end in `-sat`. The suffix marks \
         the saturating form and the non-saturating form carries no marker at \
         all; two markers or none would both mean the flag is not what \
         distinguishes them."
    );

    // -- tiebreak: shapes agreeing on m,n,k order on (a, b, c, result).
    let tuples = coop_tuples(&token_for("tiebreak"));
    assert_eq!(
        tuples.len(),
        2,
        "the tiebreak vector spells {} <coop> tuple(s), expected 2 — two \
         DISTINCT shapes agreeing on m, n and k. If they collapsed, they were \
         not distinct and the vector pins nothing about ordering.",
        tuples.len()
    );
    let field = |t: &str, i: usize| t.split('-').nth(i).unwrap_or_default().to_string();
    // m-n-k-a-b-c-result: `a` is index 3, `b` is index 4.
    assert_eq!(
        (field(&tuples[0], 3), field(&tuples[0], 4)),
        ("f16".to_string(), "f32".to_string()),
        "the first tiebreak tuple is {:?}; expected `a`=f16 and `b`=f32. The \
         two shapes SWAP `a` and `b`, so this ordering is what proves the \
         tie-break descends (a, b, c, result) rather than some other \
         permutation — under (b, a, ...) the other shape would sort first.",
        tuples[0]
    );
    assert_eq!(
        (field(&tuples[1], 3), field(&tuples[1], 4)),
        ("f32".to_string(), "f16".to_string()),
        "the second tiebreak tuple is {:?}; expected the swapped pair. Both \
         tuples must be present and in this order — checking only the first \
         would pass on a vector that dropped the second entirely.",
        tuples[1]
    );
}
