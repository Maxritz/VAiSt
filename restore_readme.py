import subprocess, sys
# get the raw blob bytes from git for origin/master:README.md
result = subprocess.run(
    ["git", "show", "origin/master:README.md"],
    capture_output=True, check=True
)
raw = result.stdout
# write as raw bytes (NOT decoded/re-encoded by Python text mode)
with open("README.md", "wb") as f:
    f.write(raw)
# verify: check for mojibake signatures
text = raw.decode("utf-8")
# em dash should be U+2014, not the cp1252-mangled '–' -> ΓÇö
if "ΓÇ" in text or "Γö" in text:
    print("WARNING: mojibake still present")
    print("sample around 'Why':", text[:500])
else:
    print("CLEAN: bytes written, no mojibake")
    print("size:", len(raw), "lines:", text.count("\n"))
    # check a few key chars
    for ch in ["\u2014", "\u2500", "\u2502", "\u251c", "\u2514", "\u2194", "\u03b1", "\u03b2"]:
        c = text.count(ch)
        if c > 0:
            print("  " + repr(ch) + " (U+%04X): %d occurrences" % ord(ch) % c)
