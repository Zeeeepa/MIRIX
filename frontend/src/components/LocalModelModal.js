import React, { useState } from 'react';
import queuedFetch from '../utils/requestQueue';
import './LocalModelModal.css';

function LocalModelModal({ isOpen, onClose, serverUrl, onSuccess }) {
  const [formData, setFormData] = useState({
    model_name: '',
    model_endpoint: '',
    temperature: 0.7,
    max_tokens: 4096,
    maximum_length: 32768
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'temperature' ? parseFloat(value) || 0 : 
              name === 'max_tokens' || name === 'maximum_length' ? parseInt(value) || 0 : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
    if (!formData.model_name.trim()) {
      setError('Model name is required');
      return;
    }
    if (!formData.model_endpoint.trim()) {
      setError('Model endpoint is required');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      const response = await queuedFetch(`${serverUrl}/models/custom/add`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          // Reset form
          setFormData({
            model_name: '',
            model_endpoint: '',
            temperature: 0.7,
            max_tokens: 4096,
            maximum_length: 32768
          });
          
          // Call success callback
          if (onSuccess) {
            onSuccess(formData.model_name);
          }
          
          onClose();
        } else {
          setError(result.message || 'Failed to add custom model');
        }
      } else {
        const errorData = await response.text();
        setError(`Failed to add custom model: ${errorData}`);
      }
    } catch (error) {
      console.error('Error adding custom model:', error);
      setError('Error adding custom model. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    // Reset form when closing
    setFormData({
      model_name: '',
      model_endpoint: '',
      temperature: 0.7,
      max_tokens: 4096,
      maximum_length: 32768
    });
    setError('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="local-model-modal">
        <div className="modal-header">
          <h3>Add Local Model</h3>
          <button className="close-button" onClick={handleClose}>
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="form-group">
            <label htmlFor="model_name">
              Model Name <span className="required">*</span>
            </label>
            <input
              type="text"
              id="model_name"
              name="model_name"
              value={formData.model_name}
              onChange={handleInputChange}
              placeholder="e.g. qwen3-32b"
              disabled={isLoading}
              required
            />
            <small className="field-description">
              The name identifier for your deployed model
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="model_endpoint">
              Model Endpoint <span className="required">*</span>
            </label>
            <input
              type="url"
              id="model_endpoint"
              name="model_endpoint"
              value={formData.model_endpoint}
              onChange={handleInputChange}
              placeholder="e.g. http://localhost:8000/v1"
              disabled={isLoading}
              required
            />
            <small className="field-description">
              The API endpoint URL for your deployed model
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="temperature">
              Temperature
            </label>
            <input
              type="number"
              id="temperature"
              name="temperature"
              value={formData.temperature}
              onChange={handleInputChange}
              min="0"
              max="2"
              step="0.1"
              disabled={isLoading}
            />
            <small className="field-description">
              Controls randomness in responses (0.0 = deterministic, 2.0 = very random)
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="max_tokens">
              Max Tokens
            </label>
            <input
              type="number"
              id="max_tokens"
              name="max_tokens"
              value={formData.max_tokens}
              onChange={handleInputChange}
              min="1"
              max="100000"
              disabled={isLoading}
            />
            <small className="field-description">
              Maximum number of tokens to generate in responses
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="maximum_length">
              Maximum Length
            </label>
            <input
              type="number"
              id="maximum_length"
              name="maximum_length"
              value={formData.maximum_length}
              onChange={handleInputChange}
              min="1"
              max="200000"
              disabled={isLoading}
            />
            <small className="field-description">
              Maximum context length supported by the model
            </small>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <div className="modal-actions">
            <button
              type="button"
              onClick={handleClose}
              className="cancel-button"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="submit-button"
              disabled={isLoading}
            >
              {isLoading ? 'Adding...' : 'Add Model'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LocalModelModal; 