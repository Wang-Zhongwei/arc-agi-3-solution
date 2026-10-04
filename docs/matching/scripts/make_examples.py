"""Render concrete matching examples; run with matplotlib installed."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'figures' / 'intro'
ROOT.mkdir(exist_ok=True)
BLUE = '#2878bc'
GREEN = '#15836d'
GRAY = '#7d8794'
INK = '#172c42'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'text.color': INK, 'axes.titlecolor': INK,
                     'savefig.facecolor': 'white'})


def save(fig, name):
    fig.savefig(ROOT / (name+'.png'), dpi=150, bbox_inches='tight')
    fig.savefig(ROOT / (name+'.svg'), bbox_inches='tight')
    plt.close(fig)


def grid(ax, array, title, labels):
    h, w = array.shape
    ax.imshow(array, cmap=ListedColormap(['#f3f6fa', BLUE]), vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which='minor', color='#dae2eb', linewidth=.55)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=15)
    for text, row, col in labels:
        ax.text(col, row, text, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':2})
    for spine in ax.spines.values():
        spine.set_visible(False)


# Example 1: two different shapes, each with one compatible destination.
fig = plt.figure(figsize=(12, 7.4))
fig.suptitle('1. Check the transform for every plausible pair', fontsize=21, fontweight='bold', y=.99)
a = np.zeros((9, 18), dtype=int)
b = a.copy()
l_shape = np.array([[1, 0], [1, 0], [1, 1]])
t_shape = np.array([[1, 1, 1], [0, 1, 0]])
a[2:5, 2:4] = l_shape
a[3:5, 12:15] = t_shape
b[2:4, 2:5] = np.rot90(l_shape, -1)
b[2:6, 10:16] = np.repeat(np.repeat(t_shape, 2, axis=0), 2, axis=1)
grid(fig.add_axes([.045,.40,.43,.46]), a, 'BEFORE', [('A', .8, 2.5), ('B', 1.5, 13)])
grid(fig.add_axes([.525,.40,.43,.46]), b, 'AFTER', [('1', .8, 3), ('2', .8, 12.5)])
ax = fig.add_axes([.18,.12,.64,.24]); ax.axis('off')
table=ax.table(cellText=[['A → 1: rotate 90°', 'A → 2: incompatible'],
                         ['B → 1: incompatible', 'B → 2: scale ×2']],
               colLabels=['After node 1', 'After node 2'], rowLabels=['Before A', 'Before B'],
               cellLoc='center', loc='center', bbox=[0,0,1,1])
table.auto_set_font_size(False); table.set_fontsize(12)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_facecolor('#e8eef5' if r==0 or c==-1 else '#f3f6fa')
    if (r,c) in ((1,0),(2,1)):
        cell.set_facecolor('#d9eee7')
fig.text(.5,.045,'Both matchers select A → 1 and B → 2: each object has only one valid destination.', ha='center', fontsize=13)
save(fig,'01_transform_candidates')


def pair_panel(ax, pairs, title, color=GREEN):
    ax.set_xlim(6.5,18); ax.set_ylim(-1.1,4.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    for y in (3,0):
        for x in range(7,18):
            ax.add_patch(Rectangle((x-.5,y-.4),1,.8,facecolor='#f3f6fa',edgecolor='#dae2eb',linewidth=.6))
    for x,y,label in ((8,3,'A'),(12,3,'B'),(11,0,'1'),(16,0,'2')):
        ax.add_patch(Rectangle((x-.44,y-.34),.88,.68,facecolor=BLUE,edgecolor='none',zorder=3))
        ax.text(x,y+(0.78 if y else -.72),label,ha='center',va='center',fontsize=14,fontweight='bold')
    for x1,x2 in pairs:
        ax.add_patch(FancyArrowPatch((x1,2.58),(x2,.44),arrowstyle='-|>',mutation_scale=15,
                                    linewidth=2,color=color,alpha=.85,connectionstyle='arc3,rad=0'))
    ax.set_xticks([8,11,12,16]);ax.set_xticklabels(['8','11','12','16'],fontsize=10)
    ax.set_xlabel('column in its frame', fontsize=11, labelpad=5)
    ax.set_yticks([0,3]);ax.set_yticklabels(['After','Before'],fontsize=11)
    ax.tick_params(axis='both',length=0)
    for spine in ax.spines.values(): spine.set_visible(False)


fig=plt.figure(figsize=(14,6.4))
fig.suptitle('2. Identical shapes create competing candidates',fontsize=22,fontweight='bold',y=.99)
fig.text(.5,.90,'Every pair is a valid move. The matcher must decide which objects belong together.',ha='center',fontsize=14)
for rect,pairs,title,color in [([.045,.40,.27,.39],[(8,11),(8,16),(12,11),(12,16)],'All 4 candidate pairs',GRAY),
                              ([.375,.40,.27,.39],[(12,11)],'Mutual nearest', '#b85c36'),
                              ([.705,.40,.27,.39],[(8,11),(12,16)],'Hungarian',GREEN)]:
    pair_panel(fig.add_axes(rect),pairs,title,color)
fig.text(.18,.21,'Distances (cells)\nA → 1: 3     A → 2: 8\nB → 1: 1     B → 2: 4',ha='center',va='center',linespacing=1.7,fontsize=12)
fig.text(.51,.21,'Keeps only B → 1 (distance 1).\nThey are each other’s nearest.\nA and node 2 stay unmatched.',ha='center',va='center',linespacing=1.7,fontsize=12)
fig.text(.84,.21,'Keeps A → 1 and B → 2.\nTotal distance = 3 + 4 = 7.\nThe other full assignment costs 9.',ha='center',va='center',linespacing=1.7,fontsize=12)
fig.text(.5,.04,'In this fixture, the true history is A → 1 and B → 2. Hungarian gets both; mutual gets one wrong pair.',ha='center',fontsize=13,fontweight='bold')
save(fig,'02_assignment_competition')

fig=plt.figure(figsize=(12,6.1))
fig.suptitle('3. The same two frames can hide different histories',fontsize=21,fontweight='bold',y=.99)
fig.text(.5,.89,'The blue pixels and their positions are identical in both panels below.',ha='center',fontsize=14)
pair_panel(fig.add_axes([.09,.32,.36,.47]),[(8,11),(12,16)],'Possible history A',GREEN)
pair_panel(fig.add_axes([.59,.32,.36,.47]),[(8,16),(12,11)],'Possible history B','#9262a8')
fig.text(.27,.19,'A moved to node 1.\nB moved to node 2.',ha='center',fontsize=13,linespacing=1.6)
fig.text(.77,.19,'A moved to node 2.\nB moved to node 1.',ha='center',fontsize=13,linespacing=1.6)
fig.text(.5,.055,'The test generator knows the history. A matcher seeing only these two frames cannot know for certain.',ha='center',fontsize=12,fontweight='bold')
save(fig,'03_identity_ambiguity')
print(ROOT)
