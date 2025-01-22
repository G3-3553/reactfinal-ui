import React from 'react';
import PropTypes from 'prop-types';

const PredictionResult = ({ prediction }) => (
  <div className="prediction-result">
    <h2>Prediction Result</h2>
    <p>{prediction}</p>
  </div>
);

PredictionResult.propTypes = {
  prediction: PropTypes.string.isRequired,
};

export default PredictionResult;
