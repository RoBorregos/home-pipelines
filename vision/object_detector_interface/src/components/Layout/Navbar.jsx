import React from 'react';

const Navbar = ({ currentPage, setCurrentPage }) => {
  return (
    <nav className="bg-black/30 backdrop-blur-sm border-b border-purple-500/20 p-4">
      <div className="max-w-7xl mx-auto flex gap-6">
        <button 
          onClick={() => setCurrentPage('dashboard')} 
          className={`${currentPage === 'dashboard' ? 'text-white font-semibold' : 'text-purple-300 hover:text-white'} transition`}
        >
          Dashboard
        </button>
        <button 
          onClick={() => setCurrentPage('run')} 
          className={`${currentPage === 'run' ? 'text-white font-semibold' : 'text-purple-300 hover:text-white'} transition`}
        >
          Run Pipeline
        </button>
        <button 
          onClick={() => setCurrentPage('past')} 
          className={`${currentPage === 'past' ? 'text-white font-semibold' : 'text-purple-300 hover:text-white'} transition`}
        >
          Past Runs
        </button>
      </div>
    </nav>
  );
};

export default Navbar;