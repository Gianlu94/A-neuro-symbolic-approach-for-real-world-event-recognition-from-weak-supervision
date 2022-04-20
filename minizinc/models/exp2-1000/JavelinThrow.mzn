% Enconding JavalinThrow in minizinc

% JT = JavelinThrow
% R  = Run
% T  = Throw

% bA = begin A     A ∈ {JT, R, T}
% eA = end A

int: bJT;
int: eJT;

int: minR = 1;
int: maxR = 83;
int: minT = 1;
int: maxT = 14;

int: targetR = maxR - minR + 1;
int: targetT = maxT - minT + 1;

% optimizer decision variables
var bJT .. eJT: bR;
var bJT .. eJT: eR;
var bJT .. eJT: bT;
var bJT .. eJT: eT;


var int: lenR = eR - bR + 1;
var int: lenT = eT - bT + 1;

% all ends have to be greater than or equal to their corresponding begin
constraint eR >= bR /\ eT >= bT;

constraint bR == bJT /\ eR == (bT-1) /\ eT == eJT;

constraint lenR >= lenT;

% component of the cost function
var int: cost_comp_run_pos = - sum (t in bR..eR) (actions_predictions[1,t]);
var int: cost_comp_run_neg = sum (t in (eR+1)..eJT) (actions_predictions[1,t]);

var int: cost_comp_throw_pos = - sum (t in bT..eT) (actions_predictions[2,t]);
var int: cost_comp_throw_neg = sum (t in bJT..(bT-1)) (actions_predictions[2,t]);

var int: cost = (
  cost_comp_run_pos + cost_comp_run_neg
  + cost_comp_throw_pos + cost_comp_throw_neg
  + 1000 * abs(targetR - lenR) + 1000 * abs(targetT - lenT)
);

solve minimize cost;