// src/Footer.js

import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p>© {new Date().getFullYear()} LegalGPT. All rights reserved.</p>

        <div className="footer-emails">
          <p>
            Contact: 
            <a href="mailto:sharmaaryan1603@gmail.com"> Aryan Sharma</a> | 
            <a href="mailto:parthsharma200428@gmail.com"> Parth Sharma</a>
          </p>
        </div>

        <p className="footer-credits">
          Made with ❤️ by the LegalGPT Team
        </p>
      </div>
    </footer>
  );
};

export default Footer;
