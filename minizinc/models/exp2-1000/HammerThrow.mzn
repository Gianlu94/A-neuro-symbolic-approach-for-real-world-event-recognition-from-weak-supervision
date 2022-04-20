% Encoding HammerThrow in minizinc

% HT_WU  = HammerThrowWindUp
% HT_S  = HammerThrowSpin
% HT_R  = HammerThrowRelease

% bA = begin A     A ∈ {HT, HT_WU, HT_S, HT_R}
% eA = end A

int: bHT;
int: eHT;

int: minHT_WU = 2;
int: maxHT_WU = 42;
int: minHT_S = 5;
int: maxHT_S = 29;
int: minHT_R = 1;
int: maxHT_R = 14;

int: targetHT_WU = maxHT_WU - minHT_WU + 1;
int: targetHT_S = maxHT_S - minHT_S + 1;
int: targetHT_R = maxHT_R - minHT_R + 1;

% optimizer decision variables
var bHT .. eHT: bHT_WU;
var bHT .. eHT: eHT_WU;
var bHT .. eHT: bHT_S;
var bHT .. eHT: eHT_S;
var bHT .. eHT: bHT_R;
var bHT .. eHT: eHT_R;

var int: lenHT_WU = eHT_WU - bHT_WU + 1;
var int: lenHT_S = eHT_S - bHT_S + 1;
var int: lenHT_R = eHT_R - bHT_R + 1;

% all ends have to be greater than their corresponding begin
constraint eHT_WU >= bHT_WU /\ eHT_S >= bHT_S /\ eHT_R >= bHT_R;

constraint bHT_WU == bHT /\ eHT_WU == (bHT_S - 1) /\ eHT_S == (bHT_R - 1) /\ eHT_R == eHT;

constraint lenHT_WU >= lenHT_R /\ lenHT_S > lenHT_R /\ lenHT_WU <= (lenHT_S + lenHT_R) /\ lenHT_S <= (lenHT_WU + lenHT_R);

% /\ lenHT_R < (lenHT_WU + len_HT_S);

% component of the cost function
var int: cost_comp_hammerthrow_wu_pos = - sum (t in bHT_WU..eHT_WU) (actions_predictions[1,t]);
var int: cost_comp_hammerthrow_wu_neg = sum (t in (eHT_WU+1)..eHT) (actions_predictions[1,t]);

var int: cost_comp_hammerthrow_s_pos = - sum (t in bHT_S..eHT_S) (actions_predictions[2,t]);
var int: cost_comp_hammerthrow_s_neg_1 = sum (t in bHT..(bHT_S-1)) (actions_predictions[2,t]);
var int: cost_comp_hammerthrow_s_neg_2 = sum (t in (eHT_S+1)..eHT) (actions_predictions[2,t]);

var int: cost_comp_hammerthrow_r_pos = - sum (t in bHT_R..eHT_R) (actions_predictions[3,t]);
var int: cost_comp_hammerthrow_r_neg = sum (t in bHT..(bHT_R-1)) (actions_predictions[3,t]);


var int: cost = (
  cost_comp_hammerthrow_wu_pos + cost_comp_hammerthrow_wu_neg
  + cost_comp_hammerthrow_s_pos + cost_comp_hammerthrow_s_neg_1 + cost_comp_hammerthrow_s_neg_2
  + cost_comp_hammerthrow_r_pos + cost_comp_hammerthrow_r_neg
  + 1000 * abs(targetHT_WU - lenHT_WU) + 1000 * abs(targetHT_S - lenHT_S) + 1000 * abs(targetHT_R -  lenHT_R)
);

solve minimize cost;