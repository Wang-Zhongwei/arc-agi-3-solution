"""Adversarial test: score every passing designer candidate on invented useful rule sentences and rewordings of the
useful lines. Sentences are invented by the critic; not drawn from any run."""
import importlib.util,sys,json
sys.dont_write_bytecode=True
ND='/Users/wang/.claude/jobs/910e124b/tmp/verify/noise/'
def load(n):
    s=importlib.util.spec_from_file_location(n,ND+n+'.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ADV=[
# clean game rules, different styles
("Confirmed: the key must be collected before the door at (12,30) opens.","gate rule w/ coord"),
("Confirmed: the counter now resets whenever the player touches a wall.","'now' rule"),
("Confirmed: each piece now occupies two cells after a merge.","'now' + rule"),
("(7,2) toggles the switch and opens the gate on the far side ✓ confirmed.","cell-subject mechanic"),
("(3,4) is a teleporter: stepping on it sends you to (9,9) ✓ confirmed.","cell-subject mechanic"),
("Confirmed: the remaining moves counter drops by 1 per action.","'remaining' rule"),
("Confirmed: the bar is at row 63 and shrinks by 2 per move.","HUD rule"),
("Win condition confirmed: every crate is at row 5.","win condition"),
("Confirmed: the exit is at (20,20) on every level.","transferable coordinate"),
("Confirmed: the goal is at the top row, cols 10-12.","goal location"),
("Confirmed: the player starts at (5,5) after every reset.","reset invariant"),
("Confirmed: the door remains open until the button is released.","'remains' rule"),
("Confirmed: blocks remain in place once they reach the floor.","'remain' rule"),
("Confirmed: the initial colour of a tile decides which button flips it.","'initial' rule"),
("Confirmed: a fresh level always starts with the budget at 64.","'fresh' rule"),
("Confirmed: the timer is still running while the menu is open.","'still' rule"),
("R = rotate clockwise, confirmed.","single-letter legend"),
("A = up, B = down, confirmed by two presses.","single-letter legend"),
("X = hazard ✓ confirmed.","single-letter legend"),
("K = key confirmed: touching it removes the lock.","single-letter legend"),
("Confirmed the movement rule:","heading (real but no content)"),
("Confirmed: pressing ACTION1 cycles colours in this order:","heading w/ claim"),
("Confirmed: the mapping is ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right.","control map"),
("Confirmed: the bar drains at step 5, 10, 15 — every 5 steps.","periodic rule citing step"),
("The enemy moves every 2 steps, confirmed at steps 4 and 6.","periodic rule citing step"),
("Confirmed: at each step the conveyor shifts the crate one cell right.","'at each step' rule"),
("At this step the rule is clear: touching lava kills ✓ confirmed.","'at this step' with rule"),
("The rule is confirmed by that: walls stop every push, and I saw it confirmed it.","pronoun"),
("Walls stop pushes, and the failed move confirmed it.","rule + bare_ref"),
("Gray cells cannot be entered; the no-op confirmed that.","rule + bare_ref"),
("Diagonal moves cost 2 units [confirmed].","rule + bracket tag"),
("Crates slide until they hit a wall [confirmed earlier].","rule + bracket tag"),
("Ice tiles keep you sliding ✓ [confirmed on level 2].","rule + bracket tag"),
("The map confirmed that doors connect opposite edges of the board.","layout word + rule"),
("The layout rule is confirmed: mirrored rooms share a single switch.","layout word + rule"),
("Confirmed geometry: pieces rotate around their centre cell.","geometry + rule"),
("The grid wraps around at the edges, confirmed by moving off the right side.","grid rule"),
("Map confirmed: the tunnels wrap from left edge to right edge.","map + rule"),
("The state resets to the start whenever a hazard is touched, confirmed.","state rule"),
("Confirmed: rows 3 and 7 are conveyor rows that carry everything right.","row-level property"),
("Confirmed: the switch at (4,4) controls both gates.","coord + control"),
("Confirmed: only tiles at even columns can be painted, e.g. col 2 and col 4.","rule + coords"),
("Confirmed: the portal at (2,2) links to the portal at (8,8).","coord link"),
("Portal pairs confirmed: (2,2) to (8,8), (3,3) to (9,9).","level fact"),
("Confirmed: the model holds, a block at (4,4) falls until it lands on a block.","model+coord rule"),
("The model is confirmed: objects at col 3 fall at twice the speed.","model+coord rule"),
("The prediction matched: a tile at (2,2) turns blue after two clicks, confirmed.","model+coord rule"),
("Confirmed: observed value at (3,3) doubles each time the lever is used.","obs+coord"),
("Confirmed: the real exit is at (10,10); the other door is a decoy.","real+coord"),
("Then the crate slides until it hits a wall, confirmed.","Then + rule"),
("Step 7 confirmed the rule: red blocks push blue blocks.","Step + rule"),
("Confirmed: score = 3 per crate, 1 per coin.","score table"),
("Costs confirmed: move=1, click=2, rotate=3.","cost table"),
("Order confirmed: A, B, C, D, A.","order list"),
("Confirmed: colours cycle R, G, B, Y, R.","cycle list"),
("Confirmed: b, N, R, Y map to buttons 1, 2, 3, 4.","list map"),
]
TARGETS={
 'cand_snapshot':{'current':['SNAP_CORE','now_coord_nocause','locative_nocause','step_strict','lead_coord'],'past':['SNAP_CORE','SNAP_PAST','now_coord_nocause','locative_nocause','coord_no_event','value_dump']},
 'cand_fragment':{'current':['core','bare_ref','letter_label','heading_colon','label_snapshot','short_verbless','core_or_snapshot','core_or_snap_or_verbless'],'past':['core','bare_ref','letter_label','label_snapshot','sym_dense','other_level_norule','core_or_sym','core_or_snapshot','core_or_snap_or_sym']},
 'cand_plan_progress':{'current':['moment','loc_strict','self_check','bare_obj','next_act','step_main','heading','union_current','union_current_no_self'],'past':['moment','loc_wide','step_any','self_check','bare_obj','layout_check','next_act','other_level','union_past','union_past_no_self']},
 'cand_past_stale_redundant':{'current':['symbol_literal','episode_trace','no_claim','trailing_colon_only','current_union'],'past':['past_union','static_coords_keeprule','static_coords','layout_noun','layout_wide','symbol_literal','episode_trace','no_claim','RED_contain_50']},
}
def line(s): return dict(text=s,next1='',level=3,prior=[])
if __name__=='__main__':
    tot={}
    for mod,phs in TARGETS.items():
        m=load(mod)
        names=sorted(set(phs['current']+phs['past']))
        for nm in names:
            if nm not in m.CANDIDATES: print('MISSING',mod,nm); continue
            fn=m.CANDIDATES[nm][1]
            hits=[s for s,_ in ADV if fn(line(s))]
            print(f"{mod[5:]}.{nm}: {len(hits)}/{len(ADV)}")
            for h in hits: print('     ',h)
