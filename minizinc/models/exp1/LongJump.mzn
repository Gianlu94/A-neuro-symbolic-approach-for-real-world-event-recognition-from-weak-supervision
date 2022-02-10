% Enconding LongJump in minizinc

% LJ = Long Jump
% R  = Run
% J  = Jump

% bA = begin A     A ∈ {HJ, R, J, F}
% eA = end A

int: bLJ;
int: eLJ;

int: avgLJ;
int: avgR;
int: avgJ;

% optimizer decision variables
var bLJ .. eLJ: bR;
var bLJ .. eLJ: eR;
var bLJ .. eLJ: bJ;
var bLJ .. eLJ: eJ;

% all ends have to be greater than or equal to their corresponding begin
constraint eR >= bR /\ eJ >= bJ;

constraint bR >= bLJ /\ eR < bJ /\ eJ <= eLJ;

% component of the cost function
var int: cost_comp_run_pos = - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg_1 = sum (t in bLJ..(bR-1)) (actions_predictions[1,t]);
var int: cost_comp_run_neg_2 = sum (t in (eR+1)..eLJ) (actions_predictions[1,t]);

var int: cost_comp_jump_pos = - sum (t in bJ..eJ) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bLJ..(bJ-1)) (actions_predictions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in (eJ+1)..eLJ) (actions_predictions[2,t]);

var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg_1 + cost_comp_run_neg_2
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + 1000 * abs(avgR - (eR - bR + 1)) + 1000 * abs(avgJ - (eJ - bJ + 1))
);

solve minimize cost;