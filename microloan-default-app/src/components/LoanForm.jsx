import React, { useState } from 'react';
import axios from 'axios';

function LoanForm() {
  const [formData, setFormData] = useState({
    monthly_income: '',
    utility_payment_timeliness: '',
    has_previous_loan: '',
    gender: '',
    loan_amount: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/predict', [formData]);
      setPrediction(response.data.predictions[0]);
    } catch (err) {
      setError("Submission failed. Please check your input.");
      console.error(err);
    }
  };

  return (
    <div className="container d-flex justify-content-center align-items-center vh-100">
      <div className="card p-4 shadow" style={{ width: '100%', maxWidth: '600px' }}>
        <h3 className="text-center mb-4">Loan Default Prediction</h3>

        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Monthly Income</label>
            <input
              type="number"
              className="form-control"
              name="monthly_income"
              value={formData.monthly_income}
              onChange={handleChange}
              required
            />
          </div>

          <div className="mb-3">
            <label className="form-label">Utility Payment Timeliness</label>
            <select
              className="form-select"
              name="utility_payment_timeliness"
              value={formData.utility_payment_timeliness}
              onChange={handleChange}
              required
            >
              <option value="">Select...</option>
              <option value="early">Early</option>
              <option value="on-time">On-Time</option>
              <option value="late">Late</option>
            </select>
          </div>

          <div className="mb-3">
            <label className="form-label">Previous Loan</label>
            <select
              className="form-select"
              name="has_previous_loan"
              value={formData.has_previous_loan}
              onChange={handleChange}
              required
            >
              <option value="">Select...</option>
              <option value={1}>Yes</option>
              <option value={0}>No</option>
            </select>
          </div>

          <div className="mb-3">
            <label className="form-label">Gender</label>
            <select
              className="form-select"
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              required
            >
              <option value="">Select...</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>

          <div className="mb-3">
            <label className="form-label">Loan Amount</label>
            <input
              type="number"
              className="form-control"
              name="loan_amount"
              value={formData.loan_amount}
              onChange={handleChange}
              required
            />
          </div>

          <button type="submit" className="btn btn-primary w-100">Predict</button>
        </form>

        {prediction !== null && (
          <div
            className={`alert mt-4 text-center fw-bold ${
              prediction === 1 ? 'alert-danger' : 'alert-success'
            }`}
            role="alert"
          >
            {prediction === 1
              ? '⚠️ High Risk: Likely to Default'
              : '✅ Low Risk: No Default Predicted'}
          </div>
        )}


        {error && (
          <div className="alert alert-danger mt-3">{error}</div>
        )}
      </div>
    </div>
  );
}

export default LoanForm;