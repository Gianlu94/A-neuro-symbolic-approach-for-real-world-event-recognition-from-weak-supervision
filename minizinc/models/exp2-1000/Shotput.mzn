% Enconding ShotPut in minizinc

% SP = ShotPut
% SPB  = ShotPutBend
% T  = Throw

% bA = begin A     A ∈ {SP, SPB, T}
% eA = end A

int: bSP;
int: eSP;

int: minSPB = 1;
int: maxSPB = 23;
int: minT = 1;
int: maxT = 34;

int: targetSPB = maxSPB - minSPB + 1;
int: targetT = maxT - minT + 1;

% optimizer decision variables
var bSP .. eSP: bSPB;
var bSP .. eSP: eSPB;
var bSP .. eSP: bT;
var bSP .. eSP: eT;

var int: lenSPB = eSPB - bSPB + 1;
var int: lenT = eT - bT + 1;

% all ends have to be greater than or equal to their corresponding begin
constraint eSPB >= bSPB /\ eT >= bT;

constraint bSPB == bSP /\ eSPB == (bT-1) /\ eT == eSP;

constraint lenSPB >= lenT;

% component of the cost function
var int: cost_comp_spb_pos = - sum (t in bSPB..eSPB) (actions_predictions[1,t]);
var int: cost_comp_spb_neg = sum (t in (eSPB+1)..eSP) (actions_predictions[1,t]);

var int: cost_comp_throw_pos = - sum (t in bT..eT) (actions_predictions[2,t]);
var int: cost_comp_throw_neg = sum (t in bSP..(bT-1)) (actions_predictions[2,t]);

var int: cost = (
  cost_comp_spb_pos + cost_comp_spb_neg
  + cost_comp_throw_pos + cost_comp_throw_neg
  + 1000 * abs(targetSPB - lenSPB) + 1000 * abs(targetT - lenT)
);

solve minimize cost;