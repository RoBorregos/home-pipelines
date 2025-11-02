import React, { useState } from 'react';
import { CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';

const StepAccordion = ({ step, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-black/30 rounded-lg p-4 border border-purple-500/10">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          {step.status === 'success' ? (
            <CheckCircle className="text-green-400" size={24} />
          ) : (
            <XCircle className="text-red-400" size={24} />
          )}
          <span className="text-white font-semibold">{step.name}</span>
        </div>
        {isExpanded ? 
          <ChevronUp className="text-purple-300" /> : 
          <ChevronDown className="text-purple-300" />
        }
      </div>
      
      {isExpanded && (
        <div className="mt-4 pl-9">
          <div className="bg-black/50 rounded p-3 font-mono text-sm text-green-300">
            {step.logs}
          </div>
        </div>
      )}
    </div>
  );
};

export default StepAccordion;