import React from 'react';

const App: React.FC = () => {
  console.log('App component rendering'); // Debug log
  
  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h1>Stock Predictor</h1>
      <p>React is working!</p>
    </div>
  );
};

export default App;