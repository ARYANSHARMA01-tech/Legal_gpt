// src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import AppRoutes from './AppRoutes'; // Handles all routing (App, Login, Signup)
import reportWebVitals from './reportWebVitals';

// Ensure there is a <div id="root"></div> in your public/index.html
const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <AppRoutes />
  </React.StrictMode>
);

// Optional: to measure performance, send results to analytics endpoint
reportWebVitals();
