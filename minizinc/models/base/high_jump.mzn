% Enconding High Jump in minizinc

% HJ = High Jump
% R  = Run
% J  = Jump
% F  = Fall

% bA = begin A     A ∈ {HJ, R, J, F}
% eA = end A

% C = clip
int: bC;
int: eC;

% optimizer decision variables
var bC .. eC: bHJ;
var bC .. eC: eHJ;
var bC .. eC: bR;
var bC .. eC: eR;
var bC .. eC: bJ;
var bC .. eC: eJ;
var bC .. eC: bF;
var bC .. eC: eF;

% all ends have to be greater than their corresponding begin
constraint eHJ > bHJ /\ eR > bR /\ eJ > bJ /\ eF > bF;

constraint bR == bHJ /\ eR < bJ /\ eJ < bF /\ eF == eHJ;

% component of the cost function
var int: cost_comp_run_pos = eR - bR - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg = sum (t in eR..eHJ) (actions_predictions[1,t]);
var int: cost_comp_jump_pos = eJ - bJ - sum (t in bJ..eJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bHJ..bJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in eJ..eHJ) (actions_predictions[2,t]);
var int: cost_comp_fall_pos = eF - bF - sum (t in bF..eF) (actions_predictions[3,t]);
var int: cost_comp_fall_neg = sum (t in bHJ..bF) (actions_predictions[3,t]);
var int: cost_comp_highjump_pos = eHJ - bHJ - sum (t in bHJ..eHJ) (actions_predictions[4,t]);
var int: cost_comp_highjump_neg_1 = sum (t in bC..bHJ) (actions_predictions[4,t]);
var int: cost_comp_highjump_neg_2 = sum (t in eHJ..eC) (actions_predictions[4,t]);


var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_fall_pos + cost_comp_fall_neg
  + cost_comp_highjump_pos + cost_comp_highjump_neg_1 + cost_comp_highjump_neg_2
);

solve minimize cost;

