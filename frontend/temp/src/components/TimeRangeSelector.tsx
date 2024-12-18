import React from 'react';

interface TimeRangeProps {
  onRangeChange: (range: string) => void;
  currentRange: string;
}

const TimeRangeSelector: React.FC<TimeRangeProps> = ({ onRangeChange, currentRange }) => {
  const ranges = ['1D', '1W', '1M', '3M', '6M', '1Y', 'ALL'];

  return (
    <div className="time-range-selector">
      {ranges.map(range => (
        <button
          key={range}
          className={currentRange === range ? 'active' : ''}
          onClick={() => onRangeChange(range)}
        >
          {range}
        </button>
      ))}
    </div>
  );
};

export default TimeRangeSelector;
