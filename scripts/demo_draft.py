""" Interactive-style demo of sequential pick recommendation.

    python scripts/demo_draft.py --radiant Pudge,Juggernaut --dire Invoker,Lion
    python scripts/demo_draft.py --radiant Pudge --dire Invoker --lookahead

Unlike upstream query.py this works at any point in the draft, not only when
nine heroes are locked.
"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import hero_names
from training.sequential_query import rank_picks, score_state

MODEL_PATH = 'artifacts/prefix_model.pkl'


def resolve(names_csv, inv):
    out = []
    for raw in [s.strip() for s in names_csv.split(',') if s.strip()]:
        match = [n for n in inv if n.lower() == raw.lower()]
        if not match:
            match = [n for n in inv if raw.lower() in n.lower()]
        if not match:
            raise SystemExit("Unknown hero: %r" % raw)
        out.append(inv[match[0]] - 1)
    return out


def main(a):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Missing %s -- run scripts/train_prefix.py first." % MODEL_PATH)
    model = pickle.load(open(MODEL_PATH, 'rb'))
    names = hero_names()
    inv = {v: k for k, v in names.items()}

    R = resolve(a.radiant, inv) if a.radiant else []
    D = resolve(a.dire, inv) if a.dire else []

    print("Radiant: %s" % (", ".join(names[h + 1] for h in R) or "(empty)"))
    print("Dire   : %s" % (", ".join(names[h + 1] for h in D) or "(empty)"))
    print("Radiant win probability in this state: %.3f" % score_state(model, R, D))

    label = "with one-ply opponent reply" if a.lookahead else "greedy"
    print("\nBest picks for %s (%s):" % (a.side, label))
    for hid, sc, nm in rank_picks(model, R, D, side=a.side, top=a.top,
                                  lookahead=a.lookahead, hero_names=names):
        print("  %-24s %.4f" % (nm, sc))

    if not a.lookahead:
        print("\nNote: greedy scores are systematically optimistic; they ignore the")
        print("opponent's counter-pick. Re-run with --lookahead to compare.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--radiant', default='')
    ap.add_argument('--dire', default='')
    ap.add_argument('--side', default='radiant', choices=['radiant', 'dire'])
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--lookahead', action='store_true')
    main(ap.parse_args())
