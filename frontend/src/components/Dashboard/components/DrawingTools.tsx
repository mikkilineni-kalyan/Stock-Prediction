import React, { useState } from 'react';
import './DrawingTools.css';

interface DrawingToolsProps {
    onToolSelect: (tool: string) => void;
    onColorSelect: (color: string) => void;
    onLineWidthSelect: (width: number) => void;
}

export const DrawingTools: React.FC<DrawingToolsProps> = ({
    onToolSelect,
    onColorSelect,
    onLineWidthSelect
}) => {
    const [selectedTool, setSelectedTool] = useState('line');
    const [selectedColor, setSelectedColor] = useState('#FF0000');
    const [lineWidth, setLineWidth] = useState(2);

    const tools = [
        { id: 'line', icon: '━', label: 'Trend Line' },
        { id: 'horizontal', icon: '―', label: 'Horizontal Line' },
        { id: 'fibonacci', icon: '⌒', label: 'Fibonacci' },
        { id: 'rectangle', icon: '□', label: 'Rectangle' },
        { id: 'channel', icon: '‖', label: 'Channel' }
    ];

    const colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', 
        '#FF00FF', '#00FFFF', '#000000', '#FFFFFF'
    ];

    const handleToolSelect = (tool: string) => {
        setSelectedTool(tool);
        onToolSelect(tool);
    };

    const handleColorSelect = (color: string) => {
        setSelectedColor(color);
        onColorSelect(color);
    };

    const handleLineWidthChange = (width: number) => {
        setLineWidth(width);
        onLineWidthSelect(width);
    };

    return (
        <div className="drawing-tools">
            <div className="tool-section">
                <h4>Drawing Tools</h4>
                <div className="tool-buttons">
                    {tools.map(tool => (
                        <button
                            key={tool.id}
                            className={`tool-button ${selectedTool === tool.id ? 'active' : ''}`}
                            onClick={() => handleToolSelect(tool.id)}
                            title={tool.label}
                        >
                            {tool.icon}
                        </button>
                    ))}
                </div>
            </div>

            <div className="color-section">
                <h4>Colors</h4>
                <div className="color-picker">
                    {colors.map(color => (
                        <div
                            key={color}
                            className={`color-option ${selectedColor === color ? 'active' : ''}`}
                            style={{ backgroundColor: color }}
                            onClick={() => handleColorSelect(color)}
                        />
                    ))}
                </div>
            </div>

            <div className="line-width-section">
                <h4>Line Width</h4>
                <input
                    type="range"
                    min="1"
                    max="5"
                    value={lineWidth}
                    onChange={(e) => handleLineWidthChange(Number(e.target.value))}
                />
            </div>
        </div>
    );
}; 