import React from 'react';
import PropTypes from 'prop-types';

const InputField = ({ label, name, onChange }) => (
  <div className="input-field">
    <label>{label}:</label>
    <input
      type="text"
      name={name}
      onChange={(e) => onChange(name, e.target.value)}
      required
    />
  </div>
);

InputField.propTypes = {
  label: PropTypes.string.isRequired,
  name: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};

export default InputField;
