import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = ({ currentPage, setCurrentPage }) => {
  return (
    <nav className="bg-black/30 backdrop-blur-sm border-b border-purple-500/20 p-4">
      <div className="max-w-7xl mx-auto flex gap-6">
        <Link to="/" className={`transition text-purple-300 hover:text-white`}>
          Dashboard
        </Link>
        <Link to="/run" className={`transition text-purple-300 hover:text-white`}>
          Run Pipeline
        </Link>
        <Link to="/past" className={`transition text-purple-300 hover:text-white`}>
          Past Runs
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;