import React from 'react';
import Navbar from '../Navbar';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Navbar />
      <div className="max-w-7xl mx-auto">
        {children}
      </div>
    </div>
  );
};

export default Layout;