import React from 'react';
import { ArrowLeft } from 'lucide-react';
import RunsTable from './RunsTable';

const PastRuns = ({ setCurrentPage, runs, setSelectedRun }) => {
  const handleViewDetails = (run) => {
    setSelectedRun(run);
    setCurrentPage('details');
  };

  return (
    <div className="mt-12 p-8">
      <button 
        onClick={() => setCurrentPage('dashboard')} 
        className="text-purple-300 hover:text-white mb-6 flex items-center gap-2"
      >
        <ArrowLeft size={20} /> Back to Dashboard
      </button>

      <h1 className="text-4xl font-bold text-white mb-8">Past Runs</h1>

      <RunsTable runs={runs} onViewDetails={handleViewDetails} />
    </div>
  );
};

export default PastRuns;