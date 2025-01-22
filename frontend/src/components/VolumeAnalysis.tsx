import React from 'react';
import Plot from 'react-plotly.js';

interface VolumeData {
  dates: string[];
  volume: number[];
  averageVolume: number;
}

const VolumeAnalysis: React.FC<{ volumeData: VolumeData }> = ({ volumeData }) => {
  return (
    <div className="volume-analysis">
      <h3>Trading Volume Analysis</h3>
      <Plot
        data={[
          {
            x: volumeData.dates,
            y: volumeData.volume,
            type: 'bar',
            name: 'Volume',
            marker: { color: '#3498db' }
          },
          {
            x: volumeData.dates,
            y: Array(volumeData.dates.length).fill(volumeData.averageVolume),
            type: 'scatter',
            name: 'Average Volume',
            line: { color: '#e74c3c', dash: 'dash' }
          }
        ]}
        layout={{
          title: 'Trading Volume',
          height: 300,
          showlegend: true,
          yaxis: { title: 'Volume' }
        }}
      />
    </div>
  );
};

export default VolumeAnalysis;
