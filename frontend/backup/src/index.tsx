import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';

console.log('Starting React application...');

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);