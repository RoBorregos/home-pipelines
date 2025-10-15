import React from 'react';
import { CheckCircle, XCircle } from 'lucide-react';

const RunsTable = ({ runs, onViewDetails }) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl overflow-hidden border border-purple-500/20">
      <table className="w-full">
        <thead className="bg-black/30">
          <tr className="text-purple-200">
            <th className="px-6 py-4 text-left">Run ID</th>
            <th className="px-6 py-4 text-left">Date</th>
            <th className="px-6 py-4 text-left">Status</th>
            <th className="px-6 py-4 text-left">Duration</th>
            <th className="px-6 py-4 text-left">Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map(run => (
            <tr key={run.id} className="border-t border-purple-500/10 hover:bg-white/5 transition">
              <td className="px-6 py-4 text-white">#{run.id}</td>
              <td className="px-6 py-4 text-purple-200">{run.date}</td>
              <td className="px-6 py-4">
                {run.status === 'success' ? (
                  <span className="flex items-center gap-2 text-green-400">
                    <CheckCircle size={16} /> Success
                  </span>
                ) : (
                  <span className="flex items-center gap-2 text-red-400">
                    <XCircle size={16} /> Failed
                  </span>
                )}
              </td>
              <td className="px-6 py-4 text-purple-200">{run.duration}</td>
              <td className="px-6 py-4">
                <button 
                  onClick={() => onViewDetails(run)}
                  className="text-purple-400 hover:text-purple-300 mr-4"
                >
                  View Details
                </button>
                <button className="text-blue-400 hover:text-blue-300">
                  Rerun
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default RunsTable;