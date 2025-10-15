import React, { useState } from 'react';
import Layout from './components/Layout/Layout';
import Dashboard from './components/Dashboard/Dashboard';
import RunPipeline from './components/RunPipeline/RunPipeline';
import PastRuns from './components/PastRuns/PastRuns';
import RunDetails from './components/RunDetails/RunDetails';

const App = () => {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [selectedRun, setSelectedRun] = useState(null);
  const [runs, setRuns] = useState([
    { 
      id: 1, 
      date: '2025-10-13 14:30', 
      status: 'success', 
      duration: '45m 23s', 
      steps: [
        { name: 'Data Loaded', status: 'success', logs: 'Successfully loaded 1,250 images from dataset' },
        { name: 'Resolution Adjustment', status: 'success', logs: 'Normalized resolution to 640x480' },
        { name: 'Silhouette Extraction', status: 'success', logs: 'Segmented 1,250 silhouettes' },
        { name: 'Background Generation', status: 'success', logs: 'Generated random backgrounds with noise' },
        { name: 'Training', status: 'success', logs: 'Training completed - mAP: 0.89' }
      ]
    },
    { 
      id: 2, 
      date: '2025-10-12 09:15', 
      status: 'failed', 
      duration: '12m 45s', 
      steps: [
        { name: 'Data Loaded', status: 'success', logs: 'Successfully loaded 980 images' },
        { name: 'Resolution Adjustment', status: 'failed', logs: 'Error: Invalid image format in file_234.jpg' }
      ]
    }
  ]);

  const handleRunComplete = () => {
    const steps = ['Data Loaded', 'Resolution Adjustment', 'Silhouette Extraction', 'Background Generation', 'Training'];
    const newRun = {
      id: runs.length + 1,
      date: new Date().toLocaleString('es-MX'),
      status: 'success',
      duration: '42m 18s',
      steps: steps.map(s => ({ name: s, status: 'success', logs: `Completed ${s}` }))
    };
    setRuns([newRun, ...runs]);
    setCurrentPage('past');
  };

  const renderPage = () => {
    switch(currentPage) {
      case 'dashboard':
        return <Dashboard setCurrentPage={setCurrentPage} runs={runs} />;
      case 'run':
        return <RunPipeline setCurrentPage={setCurrentPage} onRunComplete={handleRunComplete} />;
      case 'past':
        return <PastRuns setCurrentPage={setCurrentPage} runs={runs} setSelectedRun={setSelectedRun} />;
      case 'details':
        return <RunDetails setCurrentPage={setCurrentPage} selectedRun={selectedRun} />;
      default:
        return <Dashboard setCurrentPage={setCurrentPage} runs={runs} />;
    }
  };

  return (
    <Layout currentPage={currentPage} setCurrentPage={setCurrentPage}>
      {renderPage()}
    </Layout>
  );
};

export default App;