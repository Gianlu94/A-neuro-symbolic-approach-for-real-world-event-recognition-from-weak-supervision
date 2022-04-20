% Enconding PoleVault in minizinc

% PV = PoleVault
% R  = Run
% PVPP  = PlantVaultPlantPole
% J  = Jump
% F  = Fall

% bA = begin A     A ∈ {PV, R, PVPP, J, F}
% eA = end A

int: bPV;
int: ePV;

int: minR = 1;
int: maxR = 28;
int: minPVPP = 1;
int: maxPVPP = 5;
int: minJ = 1;
int: maxJ = 22;
int: minF = 2;
int: maxF = 17;

int: targetR = maxR - minR + 1;
int: targetPVPP = maxPVPP - minPVPP + 1;
int: targetJ =  maxJ - minJ + 1;
int: targetF = maxF - minF + 1;

% optimizer decision variables
var bPV .. ePV: bR;
var bPV .. ePV: eR;
var bPV .. ePV: bPVPP;
var bPV .. ePV: ePVPP;
var bPV .. ePV: bJ;
var bPV .. ePV: eJ;
var bPV .. ePV: bF;
var bPV .. ePV: eF;

var int: lenR = eR - bR + 1;
var int: lenPVPP = ePVPP - bPVPP + 1;
var int: lenJ = eJ - bJ + 1;
var int: lenF = eF - bF + 1;

% all ends have to be greater than or equal to their corresponding begin
constraint eR >= bR /\ ePVPP >= bPVPP /\ eJ >= bJ /\ eF >= bF;

constraint bR == bPV /\ eR == (bPVPP-1) /\ ePVPP == (bJ-1) /\ eJ == (bF-1) /\ eF == ePV;

constraint lenR >= lenPVPP /\ lenR >= lenJ /\ lenR >= lenF /\ lenPVPP <= lenJ /\ lenPVPP <= lenF /\ lenJ >= lenF;

% component of the cost function
var int: cost_comp_run_pos = - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg = sum (t in (eR+1)..ePV) (actions_predictions[1,t]);

var int: cost_comp_pvpp_pos = - sum (t in bPVPP..ePVPP) (actions_predictions[2,t]);
var int: cost_comp_pvpp_neg_1 = sum (t in bPV..(bPVPP-1)) (actions_predictions[2,t]);
var int: cost_comp_pvpp_neg_2 = sum (t in (ePVPP+1)..ePV) (actions_predictions[2,t]);

var int: cost_comp_jump_pos = - sum (t in bJ..eJ) (actions_predictions[3,t]);
var int: cost_comp_jump_neg_1 = sum (t in bPV..(bJ-1)) (actions_predictions[3,t]);
var int: cost_comp_jump_neg_2 = sum (t in (eJ+1)..ePV) (actions_predictions[3,t]);

var int: cost_comp_fall_pos = - sum (t in bF..eF) (actions_predictions[4,t]);
var int: cost_comp_fall_neg = sum (t in bPV..(bF-1)) (actions_predictions[4,t]);

var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_pvpp_pos + cost_comp_pvpp_neg_1 + cost_comp_pvpp_neg_2
  + cost_comp_jump_pos + cost_comp_jump_neg_1 + cost_comp_jump_neg_2
  + cost_comp_fall_pos + cost_comp_fall_neg
  + 1000 * abs(targetR - lenR) + 1000 * abs(targetPVPP - lenPVPP)
  + 1000 * abs(targetJ - lenJ) + 1000 * abs(targetF - lenF)
);

solve minimize cost;