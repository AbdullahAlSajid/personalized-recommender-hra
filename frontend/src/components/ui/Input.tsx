import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export const Input = ({ label, error, className = '', ...props }: InputProps) => {
  return (
    <div className="flex flex-col gap-2 w-full">
      {label && (
        <label className="text-sm font-semibold text-[#5d6875] ml-1">
          {label}
        </label>
      )}
      <input
        className={`
          w-full px-4 py-3 rounded-[16px] border border-[#e0ddd5] 
          bg-white text-[#2d3142] placeholder-[#5d6875]/50
          outline-none transition-all duration-200
          focus:border-[#4ecdc4] focus:ring-2 focus:ring-[#4ecdc4]/20
          ${error ? 'border-[#ff6b6b]' : ''}
          ${className}
        `}
        {...props}
      />
      {error && (
        <span className="text-xs text-[#ff6b6b] ml-1">{error}</span>
      )}
    </div>
  );
};
