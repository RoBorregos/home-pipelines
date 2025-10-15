import React from 'react';
import { CheckCircle } from 'lucide-react';

const ProgressTracker = ({ currentStep, steps, onCancel }) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-purple-500/20">
      <h2 className="text-2xl font-semibold text-white mb-6">Pipeline Running...</h2>
      
      <div className="mb-6">
        <div className="w-full bg-black/30 rounded-full h-4 overflow-hidden">
          <div 
            className="bg-gradient-to-r from-purple-600 to-pink-600 h-full transition-all duration-500 rounded-full"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          ></div>
        </div>
        <p className="text-purple-200 text-center mt-2">
          {Math.round(((currentStep + 1) / steps.length) * 100)}% Complete
        </p>
      </div>

      <div className="space-y-3">
        {steps.map((step, idx) => (
          <div key={idx} className="flex items-center gap-3 text-white">
            {idx < currentStep ? (
              <CheckCircle className="text-green-400" size={24} />
            ) : idx === currentStep ? (
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-purple-400 border-t-transparent"></div>
            ) : (
              <div className="w-6 h-6 rounded-full border-2 border-gray-600"></div>
            )}
            <span className={idx <= currentStep ? 'opacity-100' : 'opacity-40'}>{step}</span>
          </div>
        ))}
      </div>

      <button 
        onClick={onCancel}
        className="w-full mt-6 bg-red-600 hover:bg-red-700 text-white py-3 rounded-lg font-semibold"
      >
        Cancel Run
      </button>
    </div>
  );
};

export default ProgressTracker;