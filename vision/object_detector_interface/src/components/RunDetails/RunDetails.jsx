import React from 'react';
import { ArrowLeft, Download } from 'lucide-react';
import StepAccordion from './StepAccordion';

const RunDetails = ({ setCurrentPage, selectedRun }) => {
  if (!selectedRun) return null;

  return (
    <div className="mt-12 p-8">
      <button 
        onClick={() => setCurrentPage('past')} 
        className="text-purple-300 hover:text-white mb-6 flex items-center gap-2"
      >
        <ArrowLeft size={20} /> Back to Past Runs
      </button>

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold text-white">Run #{selectedRun.id}</h1>
          <p className="text-purple-200 mt-2">{selectedRun.date} • {selectedRun.duration}</p>
        </div>
        <button className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2">
          <Download size={20} />
          Download Results
        </button>
      </div>

      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 max-w-5xl">
        <h2 className="text-2xl font-semibold text-white mb-6">Pipeline Steps</h2>
        <div className="space-y-4">
          {selectedRun.steps.map((step, idx) => (
            <StepAccordion key={idx} step={step} index={idx} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default RunDetails;