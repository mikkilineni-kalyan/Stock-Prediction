import React from 'react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
  modelAccuracy: {
    [key: string]: number;
  };
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ 
  selectedModel, 
  onModelChange,
  modelAccuracy 
}) => {
  const models = [
    { id: 'lstm', name: 'LSTM Neural Network' },
    { id: 'transformer', name: 'Transformer Model' },
    { id: 'prophet', name: 'Prophet' },
    { id: 'xgboost', name: 'XGBoost' },
    { id: 'ensemble', name: 'Ensemble Model' },
    { id: 'hybrid', name: 'Hybrid LSTM-Transformer' }
  ];

  return (
    <div className="model-selector">
      <h3>Select AI Model</h3>
      <div className="models-grid">
        {models.map(model => (
          <div 
            key={model.id}
            className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
            onClick={() => onModelChange(model.id)}
          >
            <h4>{model.name}</h4>
            {modelAccuracy[model.id] && (
              <div className="accuracy">
                Accuracy: {modelAccuracy[model.id].toFixed(2)}%
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelSelector;
