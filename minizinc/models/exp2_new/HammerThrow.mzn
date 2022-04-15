% Encoding HammerThrow in minizinc

% HT_WU  = HammerThrowWindUp
% HT_S  = HammerThrowSpin
% HT_R  = HammerThrowRelease

% bA = begin A     A ∈ {HT, HT_WU, HT_S, HT_R}
% eA = end A

int: bHT;
int: eHT;

int: avgHT_WU;
int: avgHT_S;
int: avgHT_R;

% optimizer decision variables
var bHT .. eHT: bHT_WU;
var bHT .. eHT: eHT_WU;
var bHT .. eHT: bHT_S;
var bHT .. eHT: eHT_S;
var bHT .. eHT: bHT_R;
var bHT .. eHT: eHT_R;

% all ends have to be greater than their corresponding begin
constraint eHT_WU >= bHT_WU /\ eHT_S >= bHT_S /\ eHT_R >= bHT_R;

constraint bHT_WU == bHT /\ eHT_WU == (bHT_S - 1) /\ eHT_S == (bHT_R - 1) /\ eHT_R == eHT;

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
  + 1000 * abs(avgHT_WU - (eHT_WU - bHT_WU + 1)) + 1000 * abs(avgHT_S - (eHT_S - bHT_S + 1)) + 1000 * abs(avgHT_R -  (eHT_R - bHT_R + 1))
);

solve minimize cost;