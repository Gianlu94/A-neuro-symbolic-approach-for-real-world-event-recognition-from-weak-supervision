% Enconding ThrowDiscus in minizinc

% TD = ThrowDiscus
% TD_WU  = ThrowDiscusWindUp
% TD_R  = ThrowDiscusRelease

% bA = begin A     A ∈ {TD, TD_WU, TD_R}
% eA = end A

int: bTD;
int: eTD;

int: minTD_WU = 1;
int: maxTD_WU = 13;
int: minTD_R = 1;
int: maxTD_R = 19;

int: targetTD_WU = maxTD_WU - minTD_WU + 1;
int: targetTD_R = maxTD_R - minTD_R + 1;

% optimizer decision variables
var bTD .. eTD: bTD_WU;
var bTD .. eTD: eTD_WU;
var bTD .. eTD: bTD_R;
var bTD .. eTD: eTD_R;

var int: lenTD_WU = eTD_WU - bTD_WU + 1;
var int: lenTD_R = eTD_R - bTD_R + 1;

% all ends have to be greater than or equal to their corresponding begin
constraint eTD_WU >= bTD_WU /\ eTD_R >= bTD_R;

constraint bTD_WU == bTD /\ eTD_WU == (bTD_R-1) /\ eTD_R == eTD;

constraint lenTD_WU <= lenTD_R;


% component of the cost function
var int: cost_comp_td_wu_pos = - sum (t in bTD_WU..eTD_WU) (actions_predictions[1,t]);
var int: cost_comp_td_wu_neg = sum (t in (eTD_WU+1)..eTD) (actions_predictions[1,t]);

var int: cost_comp_td_r_pos = - sum (t in bTD_R..eTD_R) (actions_predictions[2,t]);
var int: cost_comp_td_r_neg = sum (t in bTD..(bTD_R-1)) (actions_predictions[2,t]);


var int: cost = (
    cost_comp_td_wu_pos + cost_comp_td_wu_neg
  + cost_comp_td_r_pos + cost_comp_td_r_neg
  + 1000 * abs(targetTD_WU - lenTD_WU) + 1000 * abs(targetTD_R - lenTD_R)
);

solve minimize cost;