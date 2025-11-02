import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './layout';
import RunPage from './pages/RunPage';
import PastRunsPage from './pages/PastRunsPage';
import DetailsPage from './pages/DetailsPage';
import DashboardPage from './pages/DashboardPage';

const App = () => {
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
  };

  const handleViewDetails = (run) => {
    setSelectedRun(run);
  };

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<DashboardPage runs={runs} />} />
          <Route path="/run" element={<RunPage onRunComplete={handleRunComplete} />} />
          <Route path="/past" element={<PastRunsPage runs={runs} setSelectedRun={handleViewDetails} />} />
          <Route path="/details" element={<DetailsPage selectedRun={selectedRun} />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
};

export default App;
