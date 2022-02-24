% Enconding HighJump in minizinc

% HJ = High Jump
% R  = Run
% J  = Jump
% F  = Fall

% bA = begin A     A ∈ {HJ, R, J, F}
% eA = end A

int: bHJ;
int: eHJ;

int: avgR;
int: avgJ;
int: avgF;

% optimizer decision variables
var bHJ .. eHJ: bR;
var bHJ .. eHJ: eR;
var bHJ .. eHJ: bJ;
var bHJ .. eHJ: eJ;
var bHJ .. eHJ: bF;
var bHJ .. eHJ: eF;

% all ends have to be greater than their corresponding begin
constraint eR >= bR /\ eJ >= bJ /\ eF >= bF;

constraint bR == bHJ /\ eR == (bJ-1) /\ eJ == (bF-1) /\ eF == eHJ;

% component of the cost function
var int: cost_comp_run_pos = - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg = sum (t in (eR+1)..eHJ) (actions_predictions[1,t]);

var int: cost_comp_jump_pos = - sum (t in bJ..eJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bHJ..(bJ-1)) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in (eJ+1)..eHJ) (actions_predictions[2,t]);

var int: cost_comp_fall_pos = - sum (t in bF..eF) (actions_predictions[3,t]);
var int: cost_comp_fall_neg = sum (t in bHJ..(bF-1)) (actions_predictions[3,t]);

var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_fall_pos + cost_comp_fall_neg
  + 1000 * abs(avgR - (eR - bR + 1)) + 1000 * abs(avgJ - (eJ - bJ + 1)) + 1000 * abs(avgF - (eF - bF + 1))
);

solve minimize cost;