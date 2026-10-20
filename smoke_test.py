"""Quick check: run every sample claim through parser + pipeline and print the plain-English result."""
import glob
import io
import os

from claim_parser import parse_claim_document
from pipeline import load_bundle, score_claim

bundle = load_bundle()


class Doc(io.BytesIO):
    def __init__(self, path):
        super().__init__(open(path, "rb").read())
        self.name = os.path.basename(path)


for path in sorted(glob.glob("sample_claims/*")):
    claim = parse_claim_document(Doc(path))
    print("\n##", os.path.basename(path))
    if claim["missing_required"]:
        print("   STOPPED - could not read:", claim["missing_required"])
        continue
    r = score_claim(claim, bundle)
    print(f"   {r['action_title']} ({r['level']})")
    print("   why  :", r["action_reason"])
    print("   signs:", [f["text"] for f in r["warning_signs"]])
    print("   next :", r["next_steps"])
