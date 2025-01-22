import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './ResultPage.css';

const ResultPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const { prediction } = location.state || {};

  const renderResultBox = () => {
    if (!prediction) {
      return (
        <div className="result-box poor">
          <div className="icon-container">
            <i className="fas fa-smog juggling-icon"></i>
            <i className="fas fa-industry juggling-icon"></i>
            <i className="fas fa-car juggling-icon"></i>
          </div>
          <h2 className="text-danger">No result available.</h2>
        </div>
      );
    }

    if (prediction.toLowerCase().includes('poor') || prediction.toLowerCase().includes('bad')) {
      return (
        <div className="result-box poor">
          <div className="icon-container">
            <i className="fas fa-smog juggling-icon"></i>
            <i className="fas fa-industry juggling-icon"></i>
            <i className="fas fa-car juggling-icon"></i>
          </div>
          <h2 className="text-danger">{prediction}</h2>
        </div>
      );
    }

    return (
      <div className="result-box good">
        <div className="icon-container">
          <i className="fas fa-leaf juggling-icon"></i>
          <i className="fas fa-sun juggling-icon"></i>
          <i className="fas fa-wind juggling-icon"></i>
        </div>
        <h2 className="text-success">{prediction}</h2>
      </div>
    );
  };

  return (
    <div className="result-page">
      <main className="result-container">
        <h1>Prediction Result</h1>
        {renderResultBox()}

        <div className="text-center mb-5">
          <button className="btn-warning" onClick={() => navigate('/form')}>
            Predict Again
          </button>
        </div>

        <h3 id="Bad" style={{ display: prediction?.toLowerCase().includes('poor') ? 'block' : 'none', fontSize: '1.5rem', color: 'white' }}>
          Some Ways to Improve Air Quality
        </h3>
        <div
          className="cards-container"
          id="cards-container1"
          style={{ display: prediction?.toLowerCase().includes('poor') ? 'flex' : 'none' }}
        >
          <div className="card">
            <div className="card-body">
              <i className="fas fa-tree fa-3x text-success mb-3"></i>
              <h5 className="card-title">Plant More Trees</h5>
              <p className="card-text" style={{ fontSize: '1rem' }}>Increase green cover to improve air quality and reduce pollutants.</p>
            </div>
          </div>
          <div className="card">
            <div className="card-body">
              <i className="fas fa-car fa-3x text-warning mb-3"></i>
              <h5 className="card-title">Reduce Vehicle Emissions</h5>
              <p className="card-text" style={{ fontSize: '1rem' }}>Opt for electric vehicles and public transport to minimize pollution.</p>
            </div>
          </div>
          <div className="card">
            <div className="card-body">
              <i className="fas fa-industry fa-3x text-danger mb-3"></i>
              <h5 className="card-title">Use Renewable Energy</h5>
              <p className="card-text" style={{ fontSize: '1rem' }}>Switch to clean energy sources like solar and wind to cut down emissions.</p>
            </div>
          </div>
        </div>

        <h3 id="Good" style={{ display: prediction?.toLowerCase().includes('good') ? 'block' : 'none', fontSize: '1.5rem', color: 'white' }}>
          Some Ways to Maintain Air Quality
        </h3>
        <div
          className="cards-container"
          id="cards-container2"
          style={{ display: prediction?.toLowerCase().includes('good') ? 'flex' : 'none' }}
        >
          <div className="card">
            <div className="card-body">
              <i className="fas fa-tree fa-3x text-success mb-3"></i>
              <h5 className="card-title">Promote Green Spaces</h5>
              <p className="card-text" style={{ fontSize: '1rem' }}>
                Continue planting trees and maintaining parks to absorb pollutants and produce oxygen.
              </p>
            </div>
          </div>
          <div className="card">
            <div className="card-body">
              <i className="fas fa-recycle fa-3x text-warning mb-3"></i>
              <h5 className="card-title">Reduce, Reuse, Recycle</h5>
              <p className="card-text" style={{ fontSize: '1rem' }}>Minimize waste by following the three Rs to reduce pollution from landfills.</p>
            </div>
          </div>
          <div className="card">
            <div className="card-body">
              <i className="fas fa-car fa-3x text-danger mb-3"></i>
              <h5 className="card-title">Regular Vehicle Maintenance</h5>
              <p className="card-text" style={{ fontSize: '1rem' }}>Ensure vehicles are regularly serviced to minimize emissions.</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ResultPage;
