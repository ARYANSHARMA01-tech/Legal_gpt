// src/App.js
// https://legalgpt1.onrender.com/docs

// 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
import React from 'react';
import './App.css';
import Header from './pages/Header';
import Footer from './pages/Footer';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import SessionIdPage from './pages/SessionIdPage';
import BlogPage from './pages/BlogPage'; // ✅ import this

function App() {
  return (
    <div className="app-wrapper">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/session" element={<SessionIdPage />} />
        <Route path="/blog" element={<BlogPage />} /> {/* ✅ new route */}
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
