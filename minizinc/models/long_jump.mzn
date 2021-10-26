% Enconding Long Jump in minizinc

% LJ = Long Jump
% R  = Run
% J  = Jump
% F  = Fall

% bA = begin A     A ∈ {LJ, R, J, F}
% eA = end A

int: bLJ;
int: eLJ;

% optimizer decision variables
var bLJ .. eLJ: bR;
var bLJ .. eLJ: eR;
var bLJ .. eLJ: bJ;
var bLJ .. eLJ: eJ;
var bLJ .. eLJ: bS;
var bLJ .. eLJ: eS;

% all ends have to be greater than their corresponding begin
constraint eR > bR /\ eJ > bJ /\ eS > bS;

constraint bR == bLJ /\ eR < bJ /\ eJ < bS /\ eS == eLJ;

% component of the cost function
var int: cost_comp_run_pos = eR - bR - sum (t in bR..eR) (atomic_actions[1,t]);
var int: cost_comp_run_neg = sum (t in eR..eLJ) (atomic_actions[1,t]);
var int: cost_comp_jump_pos = eJ - bJ - sum (t in bJ..eJ) (atomic_actions[2,t]);
var int: cost_comp_jump_neg_1 = sum (t in bLJ..bJ) (atomic_actions[2,t]);
var int: cost_comp_jump_neg_2 = sum (t in eJ..eLJ) (atomic_actions[2,t]);
var int: cost_comp_sit_pos = eS - bS - sum (t in bS..eS) (atomic_actions[3,t]);
var int: cost_comp_sit_neg = sum (t in bLJ..bS) (atomic_actions[3,t]);


var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_sit_pos + cost_comp_sit_neg
);

solve minimize cost;
