% Enconding Long Jump in minizinc

% LJ = Long Jump
% R  = Run
% J  = Jump
% F  = Fall

% bA = begin A     A ∈ {LJ, R, J, F}
% eA = end A

% C = clip
int: bC;
int: eC;

% optimizer decision variables
var bC .. eC: bLJ;
var bC .. eC: eLJ;
var bC .. eC: bR;
var bC .. eC: eR;
var bC .. eC: bJ;
var bC .. eC: eJ;
var bC .. eC: bS;
var bC .. eC: eS;

% all ends have to be greater than their corresponding begin
constraint eLJ > bLJ /\ eR > bR /\ eJ > bJ /\ eS > bS;

constraint bR == bLJ /\ eR < bJ /\ eJ < bS /\ eS == eLJ;

% component of the cost function
var int: cost_comp_run_pos = eR - bR - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg = sum (t in eR..eLJ) (actions_predictions[1,t]);
var int: cost_comp_jump_pos = eJ - bJ - sum (t in bJ..eJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bLJ..bJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in eJ..eLJ) (actions_predictions[2,t]);
var int: cost_comp_sit_pos = eS - bS - sum (t in bS..eS) (actions_predictions[3,t]);
var int: cost_comp_sit_neg = sum (t in bLJ..bS) (actions_predictions[3,t]);
var int: cost_comp_longjump_pos = eLJ - bLJ - sum (t in bLJ..eLJ) (actions_predictions[4,t]);
var int: cost_comp_longjump_neg_1 = sum (t in bC..bLJ) (actions_predictions[4,t]);
var int: cost_comp_longjump_neg_2 = sum (t in eLJ..eC) (actions_predictions[4,t]);


var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_sit_pos + cost_comp_sit_neg
  + cost_comp_longjump_pos + cost_comp_longjump_neg_1 + cost_comp_longjump_neg_2
);

solve minimize cost;