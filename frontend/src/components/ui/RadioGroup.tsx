import React from 'react';

interface RadioOption {
  value: string;
  label: string;
}

interface RadioGroupProps {
  name: string;
  options: RadioOption[];
  value: string;
  onChange: (value: string) => void;
  className?: string;
}

export const RadioGroup = ({ name, options, value, onChange, className = '' }: RadioGroupProps) => {
  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {options.map((option) => (
        <label 
          key={option.value} 
          className={`
            flex items-center gap-3 p-4 rounded-[16px] cursor-pointer border transition-all duration-200
            ${value === option.value 
              ? 'border-[#4ecdc4] bg-white shadow-[0_2px_8px_rgba(78,205,196,0.15)]' 
              : 'border-[#e0ddd5] bg-transparent hover:bg-white/50'}
          `}
        >
          <div className={`
            w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors
            ${value === option.value ? 'border-[#4ecdc4]' : 'border-[#e0ddd5]'}
          `}>
            {value === option.value && (
              <div className="w-2.5 h-2.5 rounded-full bg-[#4ecdc4]" />
            )}
          </div>
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={(e) => onChange(e.target.value)}
            className="hidden"
          />
          <span className="text-[#2d3142] font-medium">{option.label}</span>
        </label>
      ))}
    </div>
  );
};
