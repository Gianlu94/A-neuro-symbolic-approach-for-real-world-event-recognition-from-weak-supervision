% Enconding High Jump in minizinc

% HJ = High Jump
% R  = Run
% J  = Jump
% F  = Fall

% bA = begin A     A ∈ {HJ, R, J, F}
% eA = end A

int: bHJ;
int: eHJ;

% optimizer decision variables
var bHJ .. eHJ: bR;
var bHJ .. eHJ: eR;
var bHJ .. eHJ: bJ;
var bHJ .. eHJ: eJ;
var bHJ .. eHJ: bF;
var bHJ .. eHJ: eF;

% all ends have to be greater than their corresponding begin
constraint eR > bR /\ eJ > bJ /\ eF > bF;

constraint bR == bHJ /\ eR < bJ /\ eJ < bF /\ eF == eHJ;

% component of the cost function
var int: cost_comp_run_pos = eR - bR - sum (t in bR..eR) (atomic_actions[1,t]);
var int: cost_comp_run_neg = sum (t in eR..eHJ) (atomic_actions[1,t]);
var int: cost_comp_jump_pos = eJ - bJ - sum (t in bJ..eJ) (atomic_actions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bHJ..bJ) (atomic_actions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in eJ..eHJ) (atomic_actions[2,t]);
var int: cost_comp_fall_pos = eF - bF - sum (t in bF..eF) (atomic_actions[3,t]);
var int: cost_comp_fall_neg = sum (t in bHJ..bF) (atomic_actions[3,t]);


var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_fall_pos + cost_comp_fall_neg
);

solve minimize cost;

