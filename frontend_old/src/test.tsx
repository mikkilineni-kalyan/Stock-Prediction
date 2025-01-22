import React from 'react';
import ReactDOM from 'react-dom';

const Test = () => {
    return <h1>Test Page</h1>;
};

const root = document.getElementById('root');
if (root) {
    ReactDOM.render(<Test />, root);
} 