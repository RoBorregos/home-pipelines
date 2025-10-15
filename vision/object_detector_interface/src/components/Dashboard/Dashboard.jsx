import React from 'react';
import { Play, Clock, CheckCircle, XCircle } from 'lucide-react';

const Dashboard = ({ setCurrentPage, runs }) => {
  return (
    <div className="mt-20 p-8">
      <h1 className="text-6xl font-bold text-white mb-4 text-center">Object Detector Interface</h1>
      <p className="text-purple-200 text-center mb-12 text-lg">Vision Training System - RoBorregos @HOME</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
        <button 
          onClick={() => setCurrentPage('run')}
          className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white p-8 rounded-2xl shadow-2xl transform hover:scale-105 transition flex flex-col items-center gap-4"
        >
          <Play size={48} />
          <span className="text-2xl font-semibold">Run Pipeline</span>
          <span className="text-sm opacity-80">Start new training run</span>
        </button>
        
        <button 
          onClick={() => setCurrentPage('past')}
          className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white p-8 rounded-2xl shadow-2xl transform hover:scale-105 transition flex flex-col items-center gap-4"
        >
          <Clock size={48} />
          <span className="text-2xl font-semibold">View Past Runs</span>
          <span className="text-sm opacity-80">See training history</span>
        </button>
      </div>

      <div className="mt-12 bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 max-w-4xl mx-auto">
        <h3 className="text-white text-xl font-semibold mb-4">Recent Activity</h3>
        <div className="space-y-2">
          {runs.slice(0, 3).map(run => (
            <div key={run.id} className="flex items-center justify-between text-purple-200 py-2">
              <span>Run #{run.id}</span>
              <span className="text-sm opacity-70">{run.date}</span>
              {run.status === 'success' ? 
                <CheckCircle size={20} className="text-green-400" /> : 
                <XCircle size={20} className="text-red-400" />
              }
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;