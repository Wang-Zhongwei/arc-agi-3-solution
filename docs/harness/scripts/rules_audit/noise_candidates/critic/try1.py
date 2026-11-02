import re
NUM=re.compile(r"\d")
COORD=re.compile(r"\(\s*[-+]?\d+\s*,\s*[-+]?\d+\s*\)|\b(rows?|cols?|columns?)\s+\d+",re.I)
GENERAL=re.compile(r"→|\b(when|whenever|if|while|after|until|per|each|every|always|never|so|by)\b",re.I)
GENERAL_small=re.compile(r"→|\b(when|whenever|if|after|per|each|every|always)\b",re.I)
INPUT=re.compile(r"\b(click|press|mov|push|drag)\w*",re.I)
EVENT=re.compile(r"\b(?!confirm|verifi)\w{3,}ed\b",re.I)
RULEW=re.compile(r"\b(mechanics?|rules?)\b",re.I)
def main(s):
    p=None
    while p!=s: p,s=s,re.sub(r"\([^()]*\)|\[[^\[\]]*\]"," ",s)
    return s
def main_keep_coords(s):
    # strip asides but leave (r,c) cell pairs in place
    p=None; s=re.sub(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)",r"<\1;\2>",s)
    while p!=s: p,s=s,re.sub(r"\([^()]*\)|\[[^\[\]]*\]"," ",s)
    return s
MCOORD=re.compile(r"<[-+]?\d+;[-+]?\d+>|\b(rows?|cols?|columns?)\s+\d+",re.I)
active=lambda t:bool(GENERAL.search(t) or INPUT.search(t))
t=lambda l:l['text'].strip()
cut_off=lambda l:t(l).endswith(':')
CELL=re.compile(r"^\W*(then\s+)?\(\s*-?\d+\s*,\s*\d+\s*\)",re.I)
cell_subject=lambda l:bool(CELL.search(t(l))) and len(t(l).split())<=12
cell_nocap=lambda l:bool(CELL.search(t(l)))
MOMENT=re.compile(r"\bnow\b(?!\s+(confirmed|verified))|\bcurrently\b|\bstill\b|\bremain(s|ing)?\b|\bfresh\b|\binitial\b|\bat this (step|point|moment)\b",re.I)
LOCATIVE=re.compile(r"\b(is|are|was|were|occup(y|ies|ied))\s+(at\s+)?(rows?\s*\d|cols?\s*\d|\(\s*[-+\d])",re.I)
STEP=re.compile(r"\bat steps?\s+\d|^\W*step\s+\d+\s*:",re.I)
m_moment=lambda l:bool(MOMENT.search(main_keep_coords(t(l))))
m_loc=lambda l:bool(LOCATIVE.search(re.sub(r"<([-+]?\d+);([-+]?\d+)>",r"(\1,\2)",main_keep_coords(t(l)))))
m_step=lambda l:bool(STEP.search(main_keep_coords(t(l))))
def snapshot(l):
    return bool(NUM.search(t(l))) and not active(t(l)) and (m_moment(l) or m_loc(l) or m_step(l))
def snap_noguard(l):
    return bool(NUM.search(t(l))) and (m_moment(l) or m_loc(l) or m_step(l))
def dead_coords_a(l):
    x=t(l); return bool(COORD.search(x)) and not active(x) and not EVENT.search(x) and not RULEW.search(x)
def dead_coords_b(l):
    x=t(l); return bool(MCOORD.search(main_keep_coords(x))) and not active(x) and not EVENT.search(x) and not RULEW.search(x)
def dead_coords_c(l):
    x=t(l); return bool(COORD.search(x)) and not GENERAL_small.search(x) and not INPUT.search(x) and not EVENT.search(x) and not RULEW.search(x)
LAY=r"(state|map|geometry|layout|regions?)"
LAYOUT=re.compile(r"\b"+LAY+r"\s+(is\s+)?confirmed|\bconfirmed\s+(the\s+)?([\w-]+\s+){0,2}?"+LAY+r"\b",re.I)
layout_num=lambda l:bool(LAYOUT.search(t(l))) and bool(NUM.search(t(l)))
layout_nonum=lambda l:bool(LAYOUT.search(t(l)))
LIT=re.compile(r"(?<![\w-])[A-Z][-+]?\s*=\s*[{\d+-]|\{[^}]*[<>=]|[\"'][.A-Za-z]{2,}\s?/")
lit=lambda l:bool(LIT.search(t(l)))
frag=lambda l:cut_off(l) or cell_subject(l)
def any_of(*f): return lambda l:any(g(l) for g in f)
CANDIDATES={
 'cut_off':('both',cut_off),'cell_subject':('both',cell_subject),'cell_nocap':('both',cell_nocap),'frag':('both',frag),
 'snapshot':('both',snapshot),'snap_noguard':('both',snap_noguard),'m_moment':('both',lambda l:NUM.search(t(l)) and m_moment(l) and not active(t(l))),'m_loc':('both',lambda l:m_loc(l) and not active(t(l))),'m_step':('both',lambda l:m_step(l) and not active(t(l))),
 'dead_a':('past',dead_coords_a),'dead_b':('past',dead_coords_b),'dead_c':('past',dead_coords_c),
 'layout_num':('past',layout_num),'layout_nonum':('past',layout_nonum),'lit':('both',lit),
 'U_cur':('current',any_of(frag,snapshot)),'U_cur_lit':('current',any_of(frag,snapshot,lit)),
 'U_past_a':('past',any_of(frag,snapshot,dead_coords_a,layout_num,lit)),
 'U_past_b':('past',any_of(frag,snapshot,dead_coords_b,layout_num,lit)),
 'U_past_c':('past',any_of(frag,snapshot,dead_coords_c,layout_num,lit)),
}
