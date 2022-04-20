% Enconding LongJump in minizinc

% LJ = Long Jump
% R  = Run
% J  = Jump

% bA = begin A     A ∈ {LJ, R, J, S}
% eA = end A

int: bLJ;
int: eLJ;

int: minR = 3;
int: maxR = 29;
int: minJ = 1;
int: maxJ = 15;
int: minS = 2;
int: maxS = 6;

var int: targetR = maxR - minR + 1;
var int: targetJ = maxJ - minJ + 1;
var int: targetS = maxS - minS + 1;

% optimizer decision variables
var bLJ .. eLJ: bR;
var bLJ .. eLJ: eR;
var bLJ .. eLJ: bJ;
var bLJ .. eLJ: eJ;
var bLJ .. eLJ: bS;
var bLJ .. eLJ: eS;

var int: lenR = eR - bR + 1;
var int: lenJ = eJ - bJ + 1;
var int: lenS = eS - bS + 1;

% all ends have to be greater than or equal to their corresponding begin
constraint eR >= bR /\ eJ >= bJ /\ eS >= bS;

constraint bR == bLJ /\ eR == (bJ-1) /\ eJ == (bS-1) /\ eS == eLJ;

constraint lenR >= lenJ /\ lenR >= lenS /\ lenJ >= lenS /\ lenR >= (lenJ + lenS);

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
  + 1000 * abs(targetR - lenR) + 1000 * abs(targetJ - lenJ) + 1000 * abs(targetS - lenS)
);

solve minimize cost;