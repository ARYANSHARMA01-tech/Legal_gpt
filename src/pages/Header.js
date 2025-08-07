// src/components/Header.js

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Header.css';
import logo from '../logo.png';

const Header = () => {
  const [showMenu, setShowMenu] = useState(false);

  return (
    <nav className="top-navbar">
      <div className="nav-left">
        <img src={logo} alt="LegalGPT" className="logo-img" />
        <span className="brand-name">LegalGPT</span>
      </div>

      <div className="nav-right">
        <div
          className="nav-toggle"
          onClick={() => setShowMenu((prev) => !prev)}
          aria-label="Toggle navigation menu"
        >
          ☰
        </div>

        <div className={`nav-links ${showMenu ? 'show' : ''}`}>
          <Link to="/">
            <button>Home</button>
          </Link>
          <Link to="/session"><button>Session ID</button></Link>
          <Link to="/blog"><button>Blog</button></Link>
          {/* <button>Session ID</button> */}
        </div>
      </div>
    </nav>
  );
};

export default Header;
