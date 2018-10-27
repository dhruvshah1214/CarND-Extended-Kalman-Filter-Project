#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    assert(estimations.size() > 0);
    assert(ground_truth.size() > 0);

    int n = estimations.size();
    assert(n == ground_truth.size());

    for(int i = 0; i < estimations.size(); i++){
      VectorXd d = estimations[i] - ground_truth[i];
      d = d.array()*d.array();
      rmse += d;
    }
    rmse = (rmse / n).array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);

  //init
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  assert(!(px == 0 && py == 0));

  float rho = sqrt( px*px + py* py );
  float rho_squared = rho*rho;
  float rho_cubed = rho_squared*rho;

  //jacobian
  Hj << px/rho, py/rho, 0, 0,
      -py/rho_squared, px/rho_squared, 0, 0,
     py*( vx*py - vy*px )/rho_cubed, px*( vy*px - vx*py )/rho_cubed, px/rho, py/rho;

  return Hj;
}
