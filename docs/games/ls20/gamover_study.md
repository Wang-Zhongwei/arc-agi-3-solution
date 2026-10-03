| Game | Bar ran empty? | What the agent saw |
| --- | --- | --- |
| ka59, sc25, su15, tn36, tu93, vc33, wa30, sp80 | yes (1–3 times each) | GAME_OVER on that exact action, then an automatic RESET. The prompt says "GAME OVER occurred … automatically RESET". No silent refills. |
| ls20 | yes, 8 times | 6 silent (actions 102, 220, 332, 355, 436, 472): bar 0 → 42, lives drop by one, state=NOT_FINISHED, game_over=False. 2 reported (397, 520): only when lives reach 0. |
| bp35 | no (bar never below 38/64) | Its 3 game overs came from fatal clicks, not the bar. Same message as above. |
| ar25, cd82, ft09, lp85, r11l, s5i5 | no | — |
