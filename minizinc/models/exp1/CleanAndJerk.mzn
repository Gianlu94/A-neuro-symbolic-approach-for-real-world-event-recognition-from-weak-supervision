% Enconding CleanAndJerk in minizinc

% CJ = CleanAndJerk
% WLC  = WeigthLiftingClean
% WLJ  = WeightLiftingJerk

% bA = begin A     A ∈ {CJ, WLC, WLJ}
% eA = end A

int: bCJ;
int: eCJ;

int: avgCJ;
int: avgWLC;
int: avgWLJ;

% optimizer decision variables
var bCJ .. eCJ: bWLC;
var bCJ .. eCJ: eWLC;
var bCJ .. eCJ: bWLJ;
var bCJ .. eCJ: eWLJ;

% all ends have to be greater than or equal to their corresponding begin
constraint eWLC >= bWLC /\ eWLJ >= bWLJ ;

constraint bWLC >= bCJ /\ eWLC < bWLJ /\ eWLJ <= eCJ;

% component of the cost function
var int: cost_comp_wlc_pos = - sum (t in bWLC..eWLC) (actions_predictions[1,t]);
var int: cost_comp_wlc_neg_1 = sum (t in bCJ..(bWLC-1)) (actions_predictions[1,t]);
var int: cost_comp_wlc_neg_2 = sum (t in (eWLC+1)..eCJ) (actions_predictions[1,t]);

var int: cost_comp_wlj_pos = - sum (t in bWLJ..eWLJ) (actions_predictions[2,t]);
var int: cost_comp_wlj_neg_1 = sum (t in bCJ..(bWLJ-1)) (actions_predictions[2,t]);
var int: cost_comp_wlj_neg_2 = sum (t in (eWLJ+1)..eCJ) (actions_predictions[2,t]);


var int: cost = (
    cost_comp_wlc_pos + cost_comp_wlc_neg_1 + cost_comp_wlc_neg_2
  + cost_comp_wlj_pos + cost_comp_wlj_neg_1 + cost_comp_wlj_neg_2
  + 1000 * abs(avgWLC - (eWLC - bWLC + 1)) + 1000 * abs(avgWLJ - (eWLJ - bWLJ + 1))
);

solve minimize cost;