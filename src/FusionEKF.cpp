#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // init
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  R_laser_ << 0.0225, 0,
        0, 0.0225;

  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_[0];
  	  double phi = measurement_pack.raw_measurements_[1];
  	  double rho_dot = measurement_pack.raw_measurements_[2]; // velocity
  	  // Coordinates convertion from polar to cartesian
  	  double x = rho * cos(phi);
  	  double y = rho * sin(phi);
  	  double vx = rho_dot * cos(phi);
  	  double vy = rho_dot * sin(phi);
      ekf_.x_ << x, y, vx , vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    cout << "INIT COMPLETE" << endl;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_ ;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

   // code from lesson

   double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
   previous_timestamp_ = measurement_pack.timestamp_;

   // transition update
   ekf_.F_ = MatrixXd(4, 4);
   ekf_.F_ << 1, 0, dt, 0,
              0, 1, 0, dt,
              0, 0, 1, 0,
              0, 0, 0, 1;

   // Noise covariance
   double dt2 = dt * dt; //dt^2
   double dt3 = dt2 * dt; //dt^3
   double dt4 = dt3 * dt; //dt^4
   double dt4_4 = dt4 / 4; //dt^4/4
   double dt3_2 = dt3 / 2; //dt^3/2

   double noise_ax = 9.0;
   double noise_ay = 9.0;

   ekf_.Q_ = MatrixXd(4, 4);
   ekf_.Q_ << dt4_4 * noise_ax, 0, dt3_2 * noise_ax, 0,
              0, dt4_4 * noise_ay, 0, dt3_2 * noise_ay,
              dt3_2 * noise_ax, 0, dt2 * noise_ax, 0,
              0, dt3_2 * noise_ay, 0, dt2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

   if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
     // Radar updates
     ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
   	 ekf_.R_ = R_radar_;
   	 ekf_.UpdateEKF(measurement_pack.raw_measurements_);
   } else {
     // Laser updates
     ekf_.H_ = H_laser_;
   	 ekf_.R_ = R_laser_;
   	 ekf_.Update(measurement_pack.raw_measurements_);
   }


  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
