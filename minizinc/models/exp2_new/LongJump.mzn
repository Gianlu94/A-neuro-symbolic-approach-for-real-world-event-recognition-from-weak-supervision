% Enconding LongJump in minizinc

% LJ = Long Jump
% R  = Run
% J  = Jump

% bA = begin A     A ∈ {LJ, R, J, S}
% eA = end A

int: bLJ;
int: eLJ;

int: avgR;
int: avgJ;
int: avgS;

% optimizer decision variables
var bLJ .. eLJ: bR;
var bLJ .. eLJ: eR;
var bLJ .. eLJ: bJ;
var bLJ .. eLJ: eJ;
var bLJ .. eLJ: bS;
var bLJ .. eLJ: eS;

% all ends have to be greater than or equal to their corresponding begin
constraint eR >= bR /\ eJ >= bJ /\ eS >= bS;

constraint bR == bLJ /\ eR == (bJ-1) /\ eJ == (bS-1) /\ eS == eLJ;

% component of the cost function
var int: cost_comp_run_pos = - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg = sum (t in (eR+1)..eLJ) (actions_predictions[1,t]);

var int: cost_comp_jump_pos = - sum (t in bJ..eJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bLJ..(bJ-1)) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in (eJ+1)..eLJ) (actions_predictions[2,t]);

var int: cost_comp_sit_pos = - sum (t in bS..eS) (actions_predictions[3,t]);
var int: cost_comp_sit_neg = sum (t in bLJ..(bS-1)) (actions_predictions[3,t]);

var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_sit_pos + cost_comp_sit_neg
  + 1000 * abs(avgR - (eR - bR + 1)) + 1000 * abs(avgJ - (eJ - bJ + 1)) + 1000 * abs(avgS - (eS - bS + 1))
);

solve minimize cost;