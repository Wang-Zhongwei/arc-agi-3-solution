# What "reasoning" is, and how a local reasoning LLM should spend thinking tokens in an unfamiliar interactive game

Compiled 2026-10-08. Every claim is linked. Video links carry `&t=` offsets and the `[mm:ss]` stamp is given in the text. Transcripts were pulled with youtube-transcript-api (auto captions, so wording can be slightly off); the quoted lines are lightly de-stuttered.

Agent under study: local Qwen3.8-Flash-Next (4-bit, ~30B-class MoE reasoning model), one GPU, offline, fixed wall-clock, acts by writing Python.

## 0. How to read the evidence tiers

Per the owner's rule (2026-10-08), every harness feature in section 4 carries an evidence tier:

- **Tier 1**: measured on this model or in our runs (arms.md, research-sweep.md).
- **Tier 2**: other teams on Qwen3.x or on this same harness (Franzen's WRITEUP, competitors).
- **Tier 3**: comparable open ~30B reasoning models (QwQ-32B, R1-distill-Qwen-32B, Qwen3, s1 on Qwen2.5-32B).
- **Tier 4**: frontier models, other domains, or human cognitive science. Inspiration only. Confidence is lowered for tier-4-only items.

Almost everything in cognitive science (sections 1.2-1.9) is tier 4 by construction. The LLM-reasoning papers (section 2) are mostly tier 3 for the thinking-length results and tier 4 for the agentic/exploration results. Our own tier-1 facts are collected in section 3.

---

## 1. What reasoning consists of, per source

### 1.1 Chollet: reasoning = on-the-fly program synthesis, guided by intuition, with very little search

**Core claim.** Intelligence is "skill-acquisition efficiency over a scope of tasks, with respect to priors, experience, and generalization difficulty"; skill is the output, not the intelligence ([On the Measure of Intelligence, II.2.1](https://arxiv.org/abs/1911.01547)). Reasoning in the strong sense is not retrieving a memorized program but synthesizing a new one from parts: in the Dwarkesh interview he gives two definitions, "fetch a memorized program template and fill in values" versus "synthesize on the fly a new program based on bits and pieces of existing programs", and says the second is dramatically harder ([Dwarkesh, 21:06-21:41](https://www.youtube.com/watch?v=UakqL6Pj9xo&t=1266)). Human reasoning is always a mixture of pattern matching and discrete search ([36:55](https://www.youtube.com/watch?v=UakqL6Pj9xo&t=2215)).

**Two kinds of abstraction.** Type 1 (value-centric: perception, intuition, pattern recognition, modern ML) and type 2 (program-centric: discrete structure, exact matching, planning, reasoning). "All cognition arises from a combination of these two"; the key system-2 technique is discrete search and its blocker is combinatorial explosion ([YC "How We Get To AGI", 25:06-30:47](https://www.youtube.com/watch?v=5QcCeSsNRks&t=1506)). Chess example: intuition picks the two or three candidate moves, calculation unfolds only those ([29:47-30:17](https://www.youtube.com/watch?v=5QcCeSsNRks&t=1787)). Fixed form of the hybrid: "mostly System 2; the outer structure is a discrete program search" whose branching is tamed by learned intuition ([Dwarkesh, 51:30-52:43](https://www.youtube.com/watch?v=UakqL6Pj9xo&t=3090)).

**Humans do tiny searches.** "You're quickly building a very small set of hypotheses... maybe you have two, maybe you have three. And you test them... with backtracking... the size of the search space is tiny" ([MAD Podcast, 27:37-28:07](https://www.youtube.com/watch?v=qzb1y9gT0Sg&t=1657)). "The more sophisticated your intuition is, the fewer guesses you have to make" ([30:11](https://www.youtube.com/watch?v=qzb1y9gT0Sg&t=1811)). Efficiency, not mere success, is the definition ([MAD, 15:37](https://www.youtube.com/watch?v=qzb1y9gT0Sg&t=937)).

**Priors.** Core Knowledge = objectness (cohesion, persistence, contact), agentness/goal-directedness, number, elementary geometry/topology ([paper, III.1.2](https://arxiv.org/abs/1911.01547)).

**ARC-AGI-3.** The test is "assessing agency, the ability to explore, to learn interactively, to set goals, achieve goals autonomously" ([YC 2025, 18:50](https://www.youtube.com/watch?v=5QcCeSsNRks&t=1130)). At launch: the agent is "not told the goal or even what the controls are"; "every action you spend exploring is counted towards your efficiency score... efficiently explore, efficiently produce a world model, then infer goals, plan, and crack the game" ([YC 2026, 26:31 and 33:43](https://www.youtube.com/watch?v=k2ZLQC8P7dc&t=1591), [33:43](https://www.youtube.com/watch?v=k2ZLQC8P7dc&t=2023)). He also says the human mind "builds causal models of its surroundings as objects, agents and relations that are fundamentally symbolic and causal" ([41:01](https://www.youtube.com/watch?v=k2ZLQC8P7dc&t=2461)). On harnesses: they are valuable for automation but "if we had AGI it would make its own harness" ([25:29](https://www.youtube.com/watch?v=k2ZLQC8P7dc&t=1529)).

**What it means for the agent.**
- A thinking token is well spent when it proposes 2-3 candidate rules (program-like, checkable) or discriminates between them. Enumerating dozens of variants, or re-deriving what a cheap test could settle, is the brute-force regime he says humans avoid.
- "Intuition" (type 1) is the model's prior; the harness should supply the type-2 machinery that is cheap for code but expensive for tokens: exhaustive bookkeeping, diffs, state graphs, replay.
- Evidence tier: 4 (human cognition; his own program-synthesis systems). Confidence that "few hypotheses + test" is the right budget shape: medium.

### 1.2 Tenenbaum: reasoning = inference in a causal generative model, run by a cheap "game engine in the head"

**Core claim.** Human common sense = probabilistic inference in causally structured generative models expressed as programs; the "game engine in your head" supplies fast approximate physics and agent-planning simulators; neural nets mainly amortize the inverse (perception) step ([Tenenbaum, 18:11-19:13, 23:21](https://www.youtube.com/watch?v=90QGtYWNZPE&t=1091), [38:21 for the neural amortization role](https://www.youtube.com/watch?v=90QGtYWNZPE&t=2301)). Simulation need not be precise or long: "you only have to run a few time steps... a low precision simulation" ([36:47-37:20](https://www.youtube.com/watch?v=90QGtYWNZPE&t=2207)). The child-as-scientist idea: children "actively form hypotheses, theories, test them; sometimes play can be understood as doing experiments", and the engineering slogans are "child as coder / child as hacker" ([CVPR20, 12:03](https://www.youtube.com/watch?v=uPOpI7rMRO0&t=723), [game engine 16:47](https://www.youtube.com/watch?v=uPOpI7rMRO0&t=1007)). Foundation: [Lake, Ullman, Tenenbaum, Gershman, "Building machines that learn and think like people"](https://arxiv.org/abs/1604.00289) (causal models, intuitive physics/psychology, compositionality, learning-to-learn).

**Direct evidence on games: EMPA.** [Tsividis et al. 2021](https://arxiv.org/abs/2107.12544) (Latent Space's ARC-AGI-3 preview talk says the learning-curve chart was inspired by it, [21:44](https://www.youtube.com/watch?v=3vFu79ccDcc&t=1304)): a theory-based RL agent (EMPA) learns 90 Atari-style games in minutes, matching human learning curves, where DDQN needs 10^5-10^6 steps. Mechanics (I read the PDF):
- Hypothesis space = programs in a game-description language (objects, interaction rules, termination), scored by a Bayesian posterior; the likelihood is factored into goals, interaction events, and object dynamics.
- "Theory-based curiosity": exploratory goals are the events most informative about unknown parts of the hypothesis space, which are "always agent-object and object-object interactions for classes of objects that have not previously been observed in contact". These plans are non-myopic.
- Planning uses the current best model as a simulator, with exploration as a first-class intrinsic reward.
- Ablating theory-based exploration to epsilon-greedy "fails dramatically". Humans and EMPA are diffuse early, then targeted once object properties are known; EMPA over-explores after finding one win condition because it lacks the "games have one win condition" prior.

**What it means.**
- The unit of exploration is the object class and the interaction pair, not the raw action. "Which object types have never touched?" is the epistemic-value question for this task family.
- A world model is useful if executable/predictive; descriptive text sections are not that. This is consistent with the team's finding that structured text world-model sections did not help (and that disabling them was Franzen's "largest improvement in this area"). The harness can own the executable part (diffs, contact events, transition memory).
- Simulation depth: short and coarse is enough. This argues against long token-level mental simulation and for using the real environment (with UNDO) as the simulator.
- Evidence tier: 4 (EMPA is not an LLM), confidence medium for "object/interaction-centric exploration", low for "build an explicit Bayesian program model".

### 1.3 Schulz: children's play is experimentation chosen for information gain

**Core claims and evidence.**
- Preschoolers' play is sensitive to expected information gain. With unambiguous evidence they play indiscriminately; with ambiguous evidence they *select* (Exp. 1) and *design* (Exp. 2) interventions that isolate variables ([Cook, Goodman, Schulz 2011, Cognition](https://web.stanford.edu/~ngoodman/papers/CookGoodmanSchulz2011.pdf), abstract via [DSpace](https://dspace.mit.edu/handle/1721.1/106813)). Lecture walk-through: "only the separable pair of beads lets you isolate variables" ([CBMM talk, 25:30-28:16](https://www.youtube.com/watch?v=B6D2rgsS0-Y&t=1530)).
- They also prefer interventions whose outcomes will be *distinguishable* (shake the box that will sound different), reasoning counterfactually about their own perception *before* acting ([29:50-36:42](https://www.youtube.com/watch?v=B6D2rgsS0-Y&t=1790)).
- Infants use tiny statistical evidence to choose between two strategies: "asking for help vs exploring", depending on whether the failure is attributed to self or to the toy ([TED, 09:21-13:07](https://www.youtube.com/watch?v=y1KIVZw7Jxk&t=561)).
- Caveat she states herself: "child as scientist" is about the *structure of the representation* (coherent, causal, unobserved variables), "not about the acquisition of knowledge"; we lack a computational account of the whole exploration/inference/discovery practice ([CBMM, 03:15](https://www.youtube.com/watch?v=B6D2rgsS0-Y&t=195) and [06:55](https://www.youtube.com/watch?v=B6D2rgsS0-Y&t=415)).

**What it means.**
- Experiment choice is a first-class reasoning skill: pick the action that isolates a variable and whose outcomes differ visibly. A harness can compute "isolation" (single-factor changes) and "distinguishability" (does outcome A differ from B in the diff?) cheaply; an LLM spends thousands of tokens on this.
- When evidence is already unambiguous, extra probing is wasted ("play indiscriminately" is the sign of low information gain): this is a stop-signal for exploration.
- Evidence tier: 4. Confidence in the principle: high for humans, unknown for a 30B LLM.

### 1.4 Gopnik: explore first (high noise), then exploit; play is experimentation; empowerment as intrinsic drive

**Core claims.**
- Exploration ("figure out what the environment is like") and exploitation ("have a goal, plan, achieve it") are intrinsically in tension ([Berkeley lecture, 20:03-20:35](https://www.youtube.com/watch?v=zPbBYtwSZaU&t=1203)).
- Childhood is evolution's simulated annealing: a protected period of wide, noisy search, then narrowing ([33:11-35:21](https://www.youtube.com/watch?v=zPbBYtwSZaU&t=1991)). Children's "noise, risk taking, impulsiveness, play" are features for exploration ([34:48](https://www.youtube.com/watch?v=zPbBYtwSZaU&t=2088)).
- Blicket-detector children "wanted to play with it, try things out, experiment"; 2-year-olds' getting into everything "has the same character as scientific experiments" ([31:04-32:09](https://www.youtube.com/watch?v=zPbBYtwSZaU&t=1864)).
- Intrinsic reward = empowerment (mutual information between actions and outcomes, plus diversity of actions); learning a causal relation equals gaining control ([39:13-40:20](https://www.youtube.com/watch?v=zPbBYtwSZaU&t=2353)).
- Primary evidence: preschoolers are more willing than adolescents/adults to adopt an unusual hypothesis consistent with new evidence; adults are "less flexible" ([Gopnik et al. 2017 PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC5544286)).

**What it means.**
- Exploit mode narrows hypotheses and is *bad* at noticing the unusual explanation. A reasoning LLM that has committed to a genre/goal story is the "adult". Exploration needs structurally protected, cheap, action-heavy early turns, not long deliberation.
- Empowerment is easy to compute in a deterministic game: how many distinct outcomes the available actions produce, and which actions give controllable effects.
- Evidence tier: 4. Confidence: medium for "explore cheaply first, narrow later"; low for annealing temperature as a knob (our global temp06 arm showed no effect, tier 1).

### 1.5 Mitchell: reasoning = abstraction and analogy; accuracy can hide shortcuts

**Core claim and evidence.** Humans reason with concepts via analogy. LLMs that score well on analogy and ARC-like tasks often use surface shortcuts. In ConceptARC, asking for the *rule* showed models got correct outputs for the wrong (unintended) rule, and also had correct rules with wrong grids ([talk, 24:51-35:20](https://www.youtube.com/watch?v=X9M4HKotADo&t=1491); paper: [Beger et al. 2025](https://arxiv.org/abs/2510.02125): "the best models' rules are frequently based on surface-level shortcuts, capturing intended abstractions considerably less often than humans"). CoT traces are not always faithful to how the model solves a problem ([talk, 04:43](https://www.youtube.com/watch?v=X9M4HKotADo&t=283); see also [Chen et al. 2025, hints acknowledged in <20% of cases](https://arxiv.org/abs/2505.05410) and [Kambhampati et al., "Stop anthropomorphizing intermediate tokens"](https://arxiv.org/abs/2504.09762)). Her methodological stance is to test robustness with variations rather than accept accuracy ([19:29](https://www.youtube.com/watch?v=X9M4HKotADo&t=1169)).

**What it means.**
- Passing a level does not imply the right rule was learned (ARC Prize observed the same on ARC-AGI-3, see 2.8). The harness cannot trust the narrative in the thinking trace; verify via behavior: does the stated rule predict the next transition?
- Analogy across levels is where reasoning should pay off, but it is also where shortcuts enter (surface similarity). Provide computed structural comparisons (what objects/mechanics are new) rather than letting the model free-associate.
- Evidence tier: 4 (frontier models on ConceptARC), confidence medium.

### 1.6 Kahneman: fast coherent stories, slow effortful checking, triggered by surprise

**Core claims** (Talks at Google, [video](https://www.youtube.com/watch?v=CjVQJdIrDJ0)).
- System 2 is effortful, with limited capacity ([08:28](https://www.youtube.com/watch?v=CjVQJdIrDJ0&t=508)); System 1 builds associatively coherent interpretations and suppresses ambiguity ([32:17](https://www.youtube.com/watch?v=CjVQJdIrDJ0&t=1937)).
- A surprising event mobilizes System 2 ([33:21](https://www.youtube.com/watch?v=CjVQJdIrDJ0&t=2001)).
- Substitution: a hard question is silently answered by an easier one ([38:06](https://www.youtube.com/watch?v=CjVQJdIrDJ0&t=2286)). Confidence tracks the coherence of the story, not the amount of evidence, "so you can make a very coherent story out of very little information" ([45:50](https://www.youtube.com/watch?v=CjVQJdIrDJ0&t=2750)).
- He calls System 1/2 "fictitious characters... homunculi", a useful language, not two brain systems ([20:53](https://www.youtube.com/watch?v=CjVQJdIrDJ0&t=1253)).

**What it means for the observed failure ("misjudges goals early").** A reasoning LLM's first coherent story ("this is Sokoban/Breakout") feels confident because it is coherent. ARC Prize documents exactly this (2.8). The Kahneman-consistent remedy is not advice text but a *surprise trigger*: compare each outcome against a stored expectation and escalate deliberation only on mismatch; stay in cheap automatic mode otherwise. Evidence tier: 4. Confidence in the mapping: medium; in a specific implementation: low (see feature 6).

### 1.7 Peirce: abduction proposes, deduction predicts, induction tests; economy of research

**Core claim.** "The surprising fact, C, is observed; but if A were true, C would be a matter of course; hence there is reason to suspect A is true" ([Peirce CP 5.189 via SEP](https://plato.stanford.edu/archives/sum2026/entries/abduction/peirce.html)); abduction is "the only logical operation which introduces any new idea" (CP 5.172, same page). Inquiry loops: abduction (conjecture), deduction (derive testable consequences), induction (run the experiments), then back to abduction on failure ([SEP Peirce, section 4](https://plato.stanford.edu/entries/peirce/)). Economy of research: "proceed at every stage so as to maximize the reduction in indeterminacy of our beliefs"; research labor-time is valuable (same page).

**What it means.**
- Hypothesis formation should be *triggered* by a specific surprising observation (a diff that existing beliefs did not predict), and the output is a small set of explanations each carrying a testable consequence.
- Thinking that yields no new testable consequence is non-inquiry. This is the "reasons without acting" failure in Peircean terms: abduction without the deduction→induction step.
- Evidence tier: 4 (philosophy). Confidence: high as a description, medium as an engineering constraint.

### 1.8 Pólya: understand, plan from related problems, carry out, look back

[Four phases and heuristics](https://en.wikipedia.org/wiki/How_to_Solve_It) (secondary summary; I did not read the book): understand the problem (unknown, data, condition); plan ("do you know a related problem?", simpler problem, work backwards, figure, analogy, special cases); carry out and check each step; look back (check the result, derive it differently, see where the method applies). Pólya calls conceiving the plan "the main achievement".

**What it means.** "Related problem" = cross-level transfer; "look back" = after each level, ask why it worked. The ARC-AGI-3 failure "solved the level, didn't learn the game" (2.8) is a missing look-back. Prompt-form Pólya advice is exactly the advice-style text the team measured to hurt (tier 1); the actionable translation is a harness-computed look-back (feature 2). Tier 4, confidence medium for the structure.

### 1.9 Newell & Simon: reasoning = heuristic search over a problem space you must first construct

[Turing lecture 1976](https://home.csulb.edu/~cwallis/382/readings/482/Newell&Simonpssh.1976.html): solving = generating and modifying symbol structures until a solution test passes; a problem is a generator plus a test; "the task of intelligence is to avert the ever-present threat of the exponential explosion of search"; selectivity comes from building promise into the generator and from means-ends analysis (reduce the detected difference to the goal); "before there can be a move generator there must be a problem space". Here the problem-space (representation) is not given: in ARC-AGI-3 even the goal test is unknown, so the agent searches over *goal hypotheses* and over *state descriptions*, not only over moves.

**What it means.**
- Means-ends analysis needs a goal-difference measure. When the goal is unknown, the first job is to discover candidate difference measures (counters, HUD, matching patterns) from observed changes.
- The harness can give the model a better problem space (object table, transition graph) than raw pixels, which makes both generator and test cheaper. Tier 4; confidence medium.

### 1.10 Friston: act to resolve uncertainty first, exploit when information gain is exhausted

**Core claim.** Expected free energy decomposes into extrinsic (pragmatic) value and epistemic value (expected information gain); "epistemic value is maximized until there is no further information gain, after which exploitation is assured" ([Friston et al. 2015, Active inference and epistemic value](https://ueaeprints.uea.ac.uk/id/eprint/58264/)). In the lecture: choose the action that minimizes expected surprise, i.e. maximizes information gain about causes ([Stony Brook, 49:01-50:04](https://www.youtube.com/watch?v=G_sQZeFRjR8&t=2941)); drop the preferences and you get Bayesian surprise/infomax, put them back and you get risk-sensitive control, remove uncertainty and you get expected utility/RL ([50:35-51:36](https://www.youtube.com/watch?v=G_sQZeFRjR8&t=3035)); a never-pulled lever has the greatest "epistemic affordance" because it is novel ([56:24-57:25](https://www.youtube.com/watch?v=G_sQZeFRjR8&t=3384)); Helmholtz: "each movement we make... should be thought of as an experiment designed to test whether we have understood correctly the invariant relations" ([52:39](https://www.youtube.com/watch?v=G_sQZeFRjR8&t=3159)). (The 2015 PDF did not parse; the abstract and the lecture are what I read.)

**What it means.**
- Thinking has no epistemic value about the *world* beyond what is already in context; only observations do. The stop rule for deliberation is "expected information gain of another thought < that of a cheap action".
- A computable proxy: novelty counts (untried actions, never-contacted object pairs, un-visited states) and the recent rate of new transition types. When the novelty rate hits ~0 and the goal is still unmet, the missing knowledge is likely about the goal or a non-local mechanic, not about action effects.
- Tier 4. Confidence: medium as a design principle, low on any specific information-gain approximation.

---

## 2. LLM-reasoning research: what the evidence says about thinking tokens

### 2.1 Test-time compute scales, but with a ceiling, and the allocation matters

- Compute-optimal allocation by difficulty beats uniform best-of-N by >4x, and test-time compute can beat a 14x larger model on suitable problems ([Snell et al. 2024](https://arxiv.org/abs/2408.03314)). Tier 4.
- s1: 1K SFT examples on Qwen2.5-32B-Instruct plus **budget forcing** (cut thinking off, or append "Wait" to extend). More "Wait" helps but "does eventually flatten out at six times"; suppressing end-of-thinking too often "can lead the model into repetitive loops". Budget forcing at ~30K max "allows the model to finish with a best guess when stuck in an infinite loop" ([s1](https://arxiv.org/abs/2501.19393), PDF section 4.2 and Table 2). Tier 3.
- Qwen3 supports an explicit thinking budget: at a user threshold, insert "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>" and the model answers from partial reasoning; this "emerges naturally" from thinking-mode fusion rather than being trained ([Qwen3 technical report](https://arxiv.org/abs/2505.09388), thinking budget section). Performance scales smoothly with budget on math/coding/STEM (Qwen3-235B). Tier 3 (same family; not verified for Qwen3.8-Flash-Next).

### 2.2 Overthinking and underthinking (QwQ/R1 family, tier 3)

- Overthinking: o1-like models used 1,953% more tokens than conventional models on "2+3=?"; in >92% of cases the first solution round was already correct while taking <60% of the tokens; self-training cut tokens 48.6% on MATH500 for QwQ-32B-Preview with accuracy maintained ([Chen et al.](https://arxiv.org/abs/2412.21187)).
- Underthinking: incorrect answers use ~225% more tokens than correct ones and switch strategies more often; >70% of incorrect responses contain at least one correct thought that was abandoned; a thought-switching penalty (TIP) helps without finetuning ([Wang et al.](https://arxiv.org/abs/2501.18585); QwQ-32B-Preview, R1). So long traces are not "careful" traces; they are often unfocused ones.
- Shorter chains within a question are up to 34.5% more accurate than the longest sampled chain for the same question (LN-Super-49B, R1-Distill-32B, QwQ-32B); short-m@k voting beats majority voting; finetuning on short chains helps ([Hassid et al.](https://arxiv.org/abs/2505.17813)).
- Inverse scaling: for some tasks, longer reasoning *reduces* accuracy (distraction, overfitting to framing, spurious correlation, lost focus on deduction) ([Gema et al. 2025](https://arxiv.org/abs/2507.14417)). Tier 4 (Claude/o-series).
- Complexity collapse: LRMs beat standard models only at medium complexity, and at high complexity "reasoning effort... declines despite having remaining token budget" ([Illusion of Thinking](https://arxiv.org/abs/2506.06941)). Tier 4; contested methodology, treat as suggestive.

### 2.3 Agentic settings: interaction beats thinking, and models are bad at exploring

- **Thinking vs Doing.** Per-step budget forcing ("wait and think again") "changes the agent's actions only 15% of the time, mostly reordering subtasks rather than exploring alternatives"; best-of-n voting gives "the least improvement" despite the highest cost; scaling interaction steps has the steepest gains per token; too many re-checks "confuse" the agent; the RL-trained agent shrinks per-step reasoning as trajectories lengthen ([Shen et al., NeurIPS 2025](https://arxiv.org/abs/2506.07976), sections 4.1-4.2, Fig. 3 and 6). Tier 3 (Gemma 3 12B web agent; open, small, not Qwen).
- **Danger of overthinking.** On SWE-bench-Verified trajectories, three patterns (analysis paralysis, rogue actions, premature disengagement); overthinking correlates with lower success, more in reasoning models; picking the lower-overthinking trajectory "improves performance almost 30% while reducing cost 43%" ([Cuadron et al.](https://arxiv.org/abs/2502.08235); includes QwQ, R1-32B, o1). Tier 3 (mixed).
- **ReAct.** Interleaving reasoning with acting beats reasoning-only or acting-only: "reasoning traces help induce, track, and update action plans... actions interface with and gather additional information" ([Yao et al.](https://arxiv.org/abs/2210.03629)). Tier 4 (older models).
- **In-context exploration.** In bandits only one configuration explored well: GPT-4 + CoT + an *externally summarized* history (per-arm play counts and average reward, i.e. computed sufficient statistics; I confirmed in the PDF text) ([Krishnamurthy et al.](https://arxiv.org/abs/2403.15371)). This is evidence for *harness-computed facts* over raw history or model-written summaries. Tier 4.
- **Greediness / knowing-doing gap.** LLM agents are greedy, frequency-biased, and fail to act on what they "know" ([Schmied et al.](https://arxiv.org/abs/2504.16078)). Tier 4.
- **Experimental design.** On BoxingGym, GPT-4o struggles at designing experiments by expected information gain and at theory revision; adding an explicit statistical model did not reliably fix it ([Gandhi, Goodman et al.](https://arxiv.org/abs/2501.01540)). Tier 4.
- **AutumnBench (Tenenbaum co-author).** 43 grid-world environments, 517 humans vs five frontier models (including qwen3-235b-a22b-thinking-2507): humans win everywhere; the gap is attributed to *exploration and belief updating*. Humans use reset and no-op as experimental tools (about 12.5% of actions each; reset in every environment) while every reasoning model spends <7% on both combined (Claude 2.1%) and often never resets (Claude in 31 of 43 environments, Qwen in 8); models show "a narrow view of what counts as informative actions" and often keep using old rules after seeing contradicting evidence. More compute improved 25 of 43 environments and was flat or worse in 18 ([Warrier et al.](https://arxiv.org/abs/2510.19788), sections 4.2-5.3; text verified from the PDF). Tier 3-4 (one Qwen3-235B data point).
- **Early commitment.** "When LLMs take a wrong turn in a conversation, they get lost and do not recover": early assumptions, premature solution attempts, 39% average drop in multi-turn ([Laban et al.](https://arxiv.org/abs/2505.06120)). Tier 4; matches the observed "misjudges goals early".

### 2.4 What long reasoning actually contains (when it works)

Verification, backtracking, subgoal setting, and backward chaining are the cognitive behaviors that predict whether a model improves with RL and extra compute; priming Llama with such behaviors helped even when the examples had wrong answers ([Gandhi et al. 2025](https://arxiv.org/abs/2503.01307), Qwen2.5-3B vs Llama-3.2-3B, Countdown). Tier 3 (tiny Qwen, one puzzle). This is the LLM analogue of Peirce's loop and Pólya's look-back: the behaviors matter, length per se does not.

### 2.5 Process versus outcome

Process supervision beats outcome supervision for training a reliable math reasoner ([Lightman et al.](https://arxiv.org/abs/2305.20050), 78% on a MATH subset; PRM800K); outcome-only training matches final-answer error but leaves many flawed reasoning steps (reasoning-step error 14.0% to 3.4% only with process feedback or learned equivalents) ([Uesato et al.](https://arxiv.org/abs/2211.14275)). Tier 4; relevance: we cannot train, but a harness can supply a *process-level* check by comparing predictions with observed transitions (feature 6). Outcome-only signals (level cleared) can reinforce wrong rules (2.8).

### 2.6 Chain-of-thought is not a ground truth of what the model "believes"

Models verbalize hints they used in <20% of cases ([Chen et al.](https://arxiv.org/abs/2505.05410)); intermediate tokens are not human-like reasoning steps ([Kambhampati et al.](https://arxiv.org/abs/2504.09762)). Implication: do not build harness logic that assumes the thinking text equals the working hypothesis; prefer observable behavior (actions, predictions checked by code).

### 2.7 ARC-AGI-3 evidence specifically

- Kamradt at launch: frontier AI "has a very difficult time anticipating future events" (simulating the effect of actions 5-6 moves ahead) and "is unable to learn from its previous experiences to inform what it should be doing next. It may latch on to an early hypothesis and be unable to deviate... it thinks it's playing a different game... strong anchoring within its training set" ([launch talk, 09:53-10:55](https://www.youtube.com/watch?v=f_xT45Pi0UQ&t=593)). Needed abilities: explore, acquire own goals, build a world model on the fly, learn continuously ([12:01](https://www.youtube.com/watch?v=f_xT45Pi0UQ&t=721)). Each level should introduce a new mechanic ([Latent Space, 10:23](https://www.youtube.com/watch?v=3vFu79ccDcc&t=623)); scoring uses action efficiency vs human baselines ([technical report](https://arxiv.org/abs/2603.24621); [Latent Space 18:06](https://www.youtube.com/watch?v=3vFu79ccDcc&t=1086)).
- Preview competition: winners were not LLM-reasoners. StochasticGoose (CNN + RL predicting which actions change the frame) and Blind Squirrel (state graph, prune loops and no-change actions); LLM agents scored lower (Fluxonian DSL+LLM 8.04%, 0 games completed). Key technique: **frame-change detection** and state graphs ([30-day learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)). Graph-based explorer: median 16 levels on private games vs LLM+DSL baseline 5 levels and random 6 under 4,000 interaction limits, from "explicit graph-structured exploration, even without learning" ([Graph-Based Exploration](https://arxiv.org/abs/2512.24156)). Tier 4 for us (non-LLM or frontier), but the strongest task-specific evidence.
- Frontier post-mortem (GPT-5.5 0.43%, Opus 4.7 0.18%): (1) "true local effect, false world model": models know which action caused a change but do not turn it into a global rule; (2) wrong abstraction from training data: "a local visual resemblance becomes a full gameplay theory, then the model wastes actions testing the wrong affordances" (Tetris, Sokoban, Breakout, Flood-It...); (3) "solved the level, didn't learn the game": partial wins cement wrong theories, and "without an explicit check on why the prior level was won, models carry their misconception into the next level". Opus: "wrong compression" (latches onto false invariants); GPT-5.5: "failure to compress" (wide hypotheses, cannot commit) ([ARC Prize analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)). Tier 4 (frontier), but the failure modes match what we observe.

---

## 3. Our own tier-1 and tier-2 facts that constrain the design

From [arms.md](/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md) and [research-sweep.md](/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/research-sweep.md):

- **Helped (tier 1/2).** Animation frames/timeline as REPL variables ("strongest benefit"), `frame_diff()`, difference images, 10x upscale, persistent functions, UNDO, faithful per-action results, and **ledger-on-trim** (52.56 and 49.14 offline vs 46.25 control, 8 games, ~+4.6 mean; a memory ledger shown only right after a history trim, with confirmed/refuted tags). Pattern: harness-computed or harness-curated facts help.
- **Did not help (tier 1/2).** Structured model-written world-model sections (turning them off was Franzen's largest gain in that area); summarization (his and ours); resume-prompt tone variants; step-verification hint; thinking guide (0/28 "Predict:" lines in the old harness); scratchpad tag (model wrote zero); action-tools (called 0 times); spend-actions prompt (negative); `reasoning_effort=medium` (injects nothing in this checkpoint's template); global temperature 0.6 (no effect). Pattern: advice text and model-written structured sections are ignored or hurt; optional tool parameters get low use.
- **Inconclusive.** hypothesis-loop (`prediction` tool parameter, 93% compliance, no clear result); reasoning-summary.
- **Thinking-length facts.** Franzen yields after 2,048 generated tokens. In the control run mean output is 1,990 tokens (median 1,182, p90 4,644); 25 of 2,299 replies hit the 12k length limit, ~6.9% of generated tokens, and an 8k cap would save at most ~2.2% (research-sweep). The coordinator's figure is that 34% of tokens sit in steps of 12k+; these two are not the same quantity (a "step" apparently spans yields/resumes), and that definition should be pinned down before sizing any cap. The HF model card warns that lower effort in multi-turn agentic use "can lead to insufficient analysis, more failures, and repeated retries" (quoted in research-sweep).
- **Consensus sampling did not help**: matches Thinking-vs-Doing (best-of-n is the least efficient use of compute in agent loops) and is expensive under a fixed wall-clock.
- **Tycho port** on this model: 3.53 vs 29.22 for our harness on the same panel, because a long-frontier-model harness barely acts in 25 min on a local model (546 calls, 280 actions total): another sign that on this model the binding constraint is thinking-heavy turns per wall-clock, not harness sophistication.

---

## 4. Synthesis: where thinking tokens are worth spending

Across the sources, the same loop appears under different names: Peirce's abduction→deduction→induction, Pólya's plan→do→look back, Tenenbaum's infer-model→simulate→act, Friston's explore-until-information-exhausted→exploit, Chollet's intuition-proposes/search-verifies with 2-3 hypotheses. Four conclusions follow for a long-thinking model with a fixed wall-clock budget:

1. **Tokens are worth most where they convert a specific surprising observation into 2-3 testable hypotheses, and where they plan a multi-step action using a model that has already been tested.** Re-reading the same frame, re-deriving a settled fact, or rehearsing genre analogies have low marginal value (Chen: >92% first-solution; Wang: 70% of failed traces contained a correct thought; Cuadron: analysis paralysis).
2. **Only observations reduce uncertainty about the world (Friston); a cheap action is the way to buy one.** In interactive tasks, interaction scaling beat per-step thinking, and forced extra thinking changed the action only 15% of the time (Shen et al., tier 3).
3. **It is allocation, not "less thinking".** Underthinking (abandoning promising paths) and overthinking (re-verifying) both occur; Snell: allocate by difficulty. The high-value moments for deliberation are level start / new mechanic, a surprise (prediction miss), a contradiction, and the transition from "explored" to "exploit plan". Low-value moments are known-effect execution and repeated probing after the novelty rate has dropped.
4. **Do not trust the model's narrative or first story.** The first coherent story is anchored by training data (Kahneman; ARC Prize failure mode 2; Laban). Counter with computed facts the story must fit (what moved, what changed, what is untried), not with advice text (tier 1: advice hurts).

Caveat on all of this: almost nothing here is measured on a 4-bit ~30B MoE playing ARC-AGI-3. The cognitive-science sources are tier 4, the agentic LLM sources are mostly tier 4 or tier 3-small, and the frontier ARC-AGI-3 post-mortem may not match our model's failure modes (check against our own transcripts first; a failure-taxonomy miner is already in the research-sweep backlog).

---

## 5. Ranked harness features (8-10; ordered by expected value times confidence)

Cost estimates are per turn unless stated. "Computes" is what the harness does in code; "shows" is what enters the prompt. None relies on advice prose; each is a computed fact or a mechanical limit, per the team's finding. Confidence is after applying the tier rule: items with only tier-4 support are capped at low-medium.

### 1. Cross-turn action-effect ledger (computed): "what each action/object has been observed to do, and what is untried"
- **Computes** (in the harness, not the model): for each action (and for clicks, each clicked object class/color/size), counts tried, outcome category (no-op / small change / object moved dx,dy / HUD or counter change / level-up / death), mean changed-cell count, determinism (same outcome under same state), plus lists of untried actions and never-contacted object-class pairs. Reuse the existing frame-diff and animation machinery.
- **Shows**: a compact table (~100-250 tokens), refreshed when it changes and always after resume/trim; unchanged rows omitted. **Limits**: nothing.
- **Cost**: tokens +100-250/turn; harness CPU negligible; no extra latency.
- **Evidence**: tier 1/2: Franzen's `frame_diff`, animation timeline and faithful per-action results were his strongest perception gains; ledger-on-trim won twice; tier 4: preview winners were built on frame-change detection and state graphs ([30-day learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings), [graph exploration](https://arxiv.org/abs/2512.24156)); Krishnamurthy's computed history summary was the one setting where LLM exploration worked; AutumnBench shows LLM agents under-use resets/no-ops; ARC Prize failure mode 1 (local effects never aggregated into a rule); EMPA's object-pair novelty.
- **Confidence: medium-high** for the direction (tiers 1-2 on the underlying primitives), medium for this exact aggregated form (untested).

### 2. Level-transition briefing (computed): what was new, why the last level was won
- **Computes** at level start: (a) *delta vs earlier levels*: new object classes/colors/shapes, new HUD elements, changed action set, i.e. where the new mechanic probably is; (b) *win signature* of the previous level: the last ~5 actions, which cells/objects changed on the clearing action, which counters or matches changed, plus whether the model's stated rule (if it gave one) predicted the clear.
- **Shows**: ~150-300 tokens once at level start and once after each trim, pinned. **Limits**: none.
- **Cost**: tokens +200/level; trivial compute.
- **Evidence**: tier 1: ledger-on-trim is the one clear win and the research-sweep already proposes pinned solved-level memory; tier 4: ARC Prize failure mode 3 ("explicit check on why the prior level was won"), Pólya's look-back/related problem, Mitchell (shortcut rules), Kamradt (each level adds a mechanic), Friston/EMPA (explore where interaction types are unseen).
- **Confidence: medium.** The model-written reasoning snippet variant in the sweep is the weaker half; this version is code-computed.

### 3. Information-gated thinking cap with mechanical stop-thinking injection
- **Computes**: a per-turn thinking allowance from state: *large* at game start, level start, after a prediction/novelty miss, after a death, when the next step is a planned multi-action sequence on a validated model; *small* when the previous outcomes were all known effects or when N tokens have passed since the last tool result with no new observation. When the allowance is exceeded, inject the Qwen-style stop-thinking text and require the next output to be a tool call (a probe from list 1, or the current best action).
- **Shows**: nothing new in the prompt (a mechanical limit); optionally a one-line fact "thinking was cut at X tokens".
- **Limits**: thinking tokens per turn (suggest starting at 3-4k small / 8-12k large; tune from transcripts).
- **Cost**: negative tokens (savings); one extra prefill after injection; risk of lower quality on hard turns.
- **Evidence**: tier 3: s1 budget forcing on Qwen2.5-32B (controllable; gains flatten, loops if over-suppressed), Qwen3 thinking-budget mechanism, QwQ/R1 overthinking and underthinking results (Chen; Wang; Hassid; Cuadron); tier 3-small: interaction beat thinking in an open 12B agent (Shen). Tier 1 cautions: the research-sweep measured only ~2.2% of tokens above an 8k cap in the control run (the 34% figure needs reconciling), `reasoning_effort=medium` did nothing, and the model card warns that low effort increases retries. Franzen's 2,048-token yield exists already, so a cap must be designed with the resume path.
- **Confidence: medium** that it saves tokens; **low-medium** that it raises score (underthinking cuts the other way: gate by information, not by a flat cap). Verify Qwen3.8-Flash-Next accepts the injected stop string and what it does to MTP acceptance.

### 4. Object and contingency tracker (computed segmentation with persistent IDs)
- **Computes**: connected-component objects with IDs across frames, per-object motion, contact events, and *contingency*: which objects move when action a is taken (agent candidate), which objects move with no action (autonomous agents; test with a no-op), which act as walls/blockers/pick-ups after contact. Mask HUD bands.
- **Shows**: a short object table (~150-400 tokens) with changes since last turn only. **Limits**: none.
- **Cost**: moderate harness work; tokens +150-400.
- **Evidence**: tier 4 for the cognitive claims (Core Knowledge objectness/agentness, Tenenbaum's game-engine ontology, EMPA's object-interaction factorization); tier 4-ish for ARC-AGI-3 (graph explorer's segmentation and salience prioritization; Kamradt's objectness definition: [17:07](https://www.youtube.com/watch?v=3XmFPwjG8pg&t=1027)); tier 1: numpy-perception reached ar25 level 2 in 2/2 runs but only on 2 games on GB10 in the old harness; Franzen's perception additions helped. Counter-evidence: model-written structured text sections did not help, so this must be computed, not asked for.
- **Confidence: low-medium.**

### 5. Harness-run first look: a safe probe sweep before the first long deliberation
- **Computes/does**: at game start (and at each level start if the action set changed), the harness itself applies each available simple action once (using UNDO where reversible; test click targets only after object detection), plus one no-op, and records diffs into the ledger (feature 1). Skip actions flagged as potentially fatal.
- **Shows**: the resulting table before the model's first reasoning turn. **Limits**: probes capped (e.g., ≤ 8 + clicks ≤ 5) and only when undo works.
- **Cost**: ~10-15 game actions per game (efficiency-score cost, possibly 1-2% of a human baseline per level), ~10-20 s wall-clock; saves the long, data-free first deliberation that ends in an anchored genre guess.
- **Evidence**: tier 4 only: Peirce (abduction needs an observed surprise), Gopnik (explore before narrowing), Friston (epistemic first), Kamradt's description of humans' brief exploration then execution, preview winners' systematic action testing (non-LLM, tier 4). Tier 1 observation: the model "reasons without acting" and misjudges goals early (coordinator-reported). Risks: irreversible actions, action-efficiency penalty, deaths; the team's earlier spend-actions prompt was negative, though that was advice text.
- **Confidence: low-medium** (tier 4 only; capped). A cheap A/B on a small panel before committing.

### 6. Model-free transition-novelty flag, plus checked predictions when the model offers them
- **Computes**: for every action outcome, compare against the transition memory (feature 1): "same as before" / "new effect type" / "contradicts earlier observation (nondeterminism or hidden state)". Also compute the recent *novelty rate* (new effect types per action). If the model supplied a prediction (optional parameter), score it (cells changed, object moved).
- **Shows**: one line per action result (~20 tokens): `OUTCOME: new effect type` / `known effect`, `novelty rate over last 10 actions: 0/10`, and `PREDICTION MISS: expected …, observed …` only on miss. **Limits**: used as the gate for feature 3.
- **Cost**: +20-40 tokens/turn.
- **Evidence**: tier 4: Kahneman (surprise mobilizes System 2), Peirce (abduction triggered by the surprising fact), Friston (explore until no further information gain), Uesato/Lightman (process-level check), AutumnBench (models fail belief updating on contradiction). Tier 1 cautions: prediction parameters were low-use or inconclusive (hypothesis-loop 93% compliance but no clear result; thinking-guide 0/28), so do not rely on model-supplied predictions; the novelty flag needs none.
- **Confidence: low-medium.**

### 7. Hypothesis cap with a falsifier (finish and re-test the existing hypothesis-loop arm)
- **Computes**: the `prediction`/`hypothesis` parameter limited to ≤3 candidates, each with an action that would falsify it; the harness stores them, checks the next outcomes via feature 6, and marks confirmed/refuted in the existing ledger.
- **Shows**: the ledger lines (already part of ledger-on-trim). **Limits**: ≤3 open hypotheses.
- **Cost**: ~50-150 tokens per turn; model compliance is the risk.
- **Evidence**: tier 4: Chollet ("two or three" hypotheses), Peirce, Polya; tier 1: hypothesis-loop inconclusive, user's own ledger idea ("a ledger that keeps track of failed and successful hypotheses") half-present in ledger-on-trim; tier 4 counterpoint: Gandhi (behaviors, not length, matter; Qwen2.5-3B).
- **Confidence: low.**

### 8. Ranked experiment menu from novelty counts (not advice, just candidates)
- **Computes**: candidate probes ordered by novelty: untried actions in the current state, never-contacted object pairs, reset/no-op/undo tests, single-factor changes versus the last state (isolation) and outcome distinguishability from earlier probes.
- **Shows**: top 3 as facts ("never observed: contact between object type A and B; ACTION4 untried in this state"). **Limits**: none.
- **Cost**: +60-100 tokens/turn.
- **Evidence**: tier 4 only (Cook/Goodman/Schulz, Friston, EMPA, AutumnBench, BoxingGym). Risk: reads like advice and may be ignored or hurt as advice text did (tier 1). Mostly redundant with the "untried" part of feature 1; implement it only as part of that table.
- **Confidence: low** (capped).

### 9. Per-level temperature annealing (explore hot, exploit cold)
- **Computes**: sampling temperature scheduled by the novelty rate, high at level start, lower once the plan is validated.
- **Cost**: free. **Evidence**: tier 4 (Gopnik's annealing analogy); tier 1 negative: a global 0.6 arm had no effect, and Qwen recommends T=1.0 for thinking mode. **Confidence: low.** Worth only as a free riser on another arm.

### 10. Replay-checked rule verification (process-supervision analogue)
- **Computes**: when the model states a rule ("ACTION3 moves the player left 1"), replay from a saved state via UNDO and compare.
- **Cost**: actions and complexity. **Evidence**: tier 4 (Lightman, Uesato). Tier 1/2: the research-sweep already judged world-model and executable-hypothesis kits "a poor use" of limited days and non-transferring from frontier models; Tycho's executable world model never fired on this model. **Confidence: low. Not recommended** unless features 1-6 land and leave time.

### Suggested testing order given the "5 features at a time" workflow
Features 1, 2, 6 (cheap, additive, one shared data structure), then 3 gated by 6, then 4 and 5 as separate screens. Per the repeated-run rule, each is a screen until repeats agree. Always measure: tokens per level, % of tokens in steps ≥ 12k (with a fixed definition), actions per level, and mean `novelty rate` at clear.

---

## 6. What the sources say not to do (consistent with the team's dead ends)

- Advice-in-prompt versions of Pólya/Kahneman ("consider related problems", "slow down on surprise"): no source shows that text advice changes a local model's allocation; our tier-1 data show it does not or hurts. Convert to computed facts and mechanical limits.
- Model-written structured world models: not what Tenenbaum/EMPA or the preview winners used (executable, harness-held); consistent with our negative result.
- Multi-sample consensus per step: least efficient use of compute in agent loops (Shen et al.), plus a fixed wall-clock.
- Flat thinking-length caps: underthinking shows that long, switching traces are bad, but short is not automatically good (s1: more "Wait" helped before flattening; ARC Prize: Opus "wrong compression" vs GPT "failure to compress"). Gate by information, not by length alone.
- Summaries written by the model: the one LLM-exploration result that worked used computed statistics (Krishnamurthy), not generated summaries.

## 7. Gaps and caveats

- No source measures a 4-bit ~30B MoE agent on ARC-AGI-3; the nearest are QwQ/R1-distill-32B on static math (tier 3), a Gemma-3-12B web agent, and one Qwen3-235B data point in AutumnBench. All feature confidences are therefore low-to-medium.
- Primary texts not read directly: Peirce's Collected Papers (quoted via SEP), Pólya's book (via Wikipedia summary), Kahneman's book (talk only), the Friston 2015 PDF (parse failed; abstract and lecture read), Schulz and Bonawitz 2007 and Gopnik's causal-learning reviews (only Cook et al. and PNAS 2017). Dual-process critiques (Evans and Stanovich) were not covered; Kahneman himself calls the two systems fictions.
- Transcripts are auto-captions; no Chollet talk specifically on "reasoning in ARC-AGI-3 agents" exists beyond the launch material; the Dwarkesh 2024 interview and the YC talks carry the reasoning content.
- Open question needing our data: the definition of a "step" behind "34% of tokens in 12k+ steps" versus the sweep's 6.9% of tokens in length-truncated replies; and whether long steps contain any new observation between consecutive tool results (that single measurement would size feature 3).
- Unverified assumption: that Qwen3.8-Flash-Next honors Qwen3's inserted stop-thinking string and that UNDO is available in enough games for feature 5.

## 8. Source index (all fetched or read this session)

Videos (transcripts): [Kamradt, ARC-AGI-3 launch](https://www.youtube.com/watch?v=f_xT45Pi0UQ); [Kamradt, AI Engineer preview](https://www.youtube.com/watch?v=3XmFPwjG8pg); [Latent Space ARC-AGI-3](https://www.youtube.com/watch?v=3vFu79ccDcc); [Chollet YC 2026](https://www.youtube.com/watch?v=k2ZLQC8P7dc); [Chollet YC 2025](https://www.youtube.com/watch?v=5QcCeSsNRks); [Chollet-Dwarkesh](https://www.youtube.com/watch?v=UakqL6Pj9xo); [Chollet MAD podcast](https://www.youtube.com/watch?v=qzb1y9gT0Sg); [Tenenbaum CEU](https://www.youtube.com/watch?v=90QGtYWNZPE); [Tenenbaum CVPR20](https://www.youtube.com/watch?v=uPOpI7rMRO0); [Schulz CBMM](https://www.youtube.com/watch?v=B6D2rgsS0-Y); [Schulz TED](https://www.youtube.com/watch?v=y1KIVZw7Jxk); [Gopnik Berkeley](https://www.youtube.com/watch?v=zPbBYtwSZaU); [Mitchell](https://www.youtube.com/watch?v=X9M4HKotADo); [Kahneman Talks at Google](https://www.youtube.com/watch?v=CjVQJdIrDJ0); [Friston Stony Brook](https://www.youtube.com/watch?v=G_sQZeFRjR8). Not used (checked, redundant): Chollet-Altman fireside ([XUu-i9Wbh-c](https://www.youtube.com/watch?v=XUu-i9Wbh-c); only the minimal-program remark at 15:34), Lex Fridman #120, Mindscape 280.

Transcript files for re-checking: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/yt/<videoId>.txt`; extracted PDFs in `.../scratchpad/pdf/`.
