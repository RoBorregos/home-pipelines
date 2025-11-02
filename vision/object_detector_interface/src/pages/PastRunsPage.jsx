import React from 'react';
import { ArrowLeft } from 'lucide-react';
import RunsTable from '../_components/RunsTable';
import { useNavigate } from 'react-router-dom';

const PastRunsPage = ({ runs, setSelectedRun }) => {
  const navigate = useNavigate();

  const handleViewDetails = (run) => {
    setSelectedRun && setSelectedRun(run);
    navigate('/details');
  };

  return (
    <div className="mt-12 p-8">
      <button 
        onClick={() => navigate('/')} 
        className="text-purple-300 hover:text-white mb-6 flex items-center gap-2"
      >
        <ArrowLeft size={20} /> Back to Dashboard
      </button>

      <h1 className="text-4xl font-bold text-white mb-8">Past Runs</h1>

      <RunsTable runs={runs} onViewDetails={handleViewDetails} />
    </div>
  );
};

export default PastRunsPage;
