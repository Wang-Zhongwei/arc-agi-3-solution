import re
HEADER=('The code executed','Executed actions:','You are still on the same level.','Current state:','Valid actions right now:','Your last python snippet executed NO','For reference, the most recent','Your previous exchange this turn','No previous sequence','The code executed')
INSTR=('Only tool:','Only letter-coded','Keep tool output compact','For the most recent change, compare','Use Python to inspect','Focus on what changed','When ready, call','You may call `action','When calling `python`','If you use MOUSE')
LVL=('You have completed the previous level','Start from the mechanics','Reassess the goal','Before you study the new board','Your retained functions remain available from previous level')
RET=('Your retained functions from earlier Python calls',)
IMG=('Current grid image','Diff image')
ANIM=('The last executed action (`','Animation digest','(+','Trails are heuristic','- ','  - ','The last executed actions')
def classify_turn(text):
    """-> dict seg -> chars, and list of (seg,line)"""
    out={}; lines=[]
    blk=None
    for line in text.split('\n'):
        if line in ('<this_level>','<cleared_level>'):
            blk=line[1:-1]; out[blk]=out.get(blk,0)+len(line)+1; lines.append((blk,line)); continue
        if blk:
            out[blk]=out.get(blk,0)+len(line)+1; lines.append((blk,line))
            if line in ('</this_level>','</cleared_level>'): blk=None
            continue
        if not line.strip(): seg='blank'
        elif line.startswith(HEADER): seg='header'
        elif line.startswith(INSTR): seg='instr'
        elif line.startswith(LVL): seg='levelchange'
        elif line.startswith(RET): seg='retained'
        elif line.startswith(IMG): seg='img'
        elif line.startswith(ANIM): seg='anim'
        else: seg='other'
        out[seg]=out.get(seg,0)+len(line)+1; lines.append((seg,line))
    return out,lines
