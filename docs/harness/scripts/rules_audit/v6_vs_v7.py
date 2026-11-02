import re, sys, glob, os, json, importlib.util, collections as C
def load(path, name):
    spec=importlib.util.spec_from_file_location(name, path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
import subprocess, tempfile
from audit_paths import REPO
def at_commit(commit, name):
    # v6 and v7 are read from Git so later edits to confirmed_rules.py do not change this comparison
    src = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{commit}:ARC3-Inference/inference/agent/confirmed_rules.py'])
    tmp = tempfile.NamedTemporaryFile(suffix='.py', delete=False); tmp.write(src); tmp.close()
    return load(tmp.name, name)
v6=at_commit('0b2b66d8','cr6')
v7=at_commit('a71991a2','cr7')
run=sys.argv[1]
HDR=re.compile(r'^(\[(THINKING|TOOL CALL: \w+|TOOL RESULT: \w+|ASSISTANT|ANALYZER STATUS|USER PROMPT|SYSTEM PROMPT|MODEL RESPONSE META)\]|--- analysis_step=(\d+).*---)\s*$')
cnt=C.Counter(); only6=[]
for f in sorted(glob.glob(os.path.join(run,'transcripts','*.txt'))):
    g=os.path.basename(f).split('-')[0]; sec=None; buf=[]
    def flush():
        if sec!='THINKING': return
        t='\n'.join(buf)
        a=v6.classified_sentences(t,confirmed=True,refuted=True); b=v7.classified_sentences(t,confirmed=True,refuted=True)
        for k,_ in a: cnt['v6 '+k]+=1
        for k,_ in b: cnt['v7 '+k]+=1
        bs={s for _,s in b}
        for k,s in a:
            if s not in bs and not any(s[:40] in x or x[:40] in s for x in bs): only6.append((g,k,s))
    for line in open(f,encoding='utf-8',errors='replace'):
        m=HDR.match(line.rstrip('\n'))
        if m: flush(); buf=[]; sec=m.group(2)
        else: buf.append(line.rstrip('\n'))
    flush()
print(dict(cnt)); print('v6-only sentences', len(only6), C.Counter(k for _,k,_ in only6))
import random; random.seed(1)
for g,k,s in random.sample(only6, min(25,len(only6))): print('-',g,k,'|',s[:140])
